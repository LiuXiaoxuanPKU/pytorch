#include <torch/csrc/python_headers.h>

#include <pybind11/pybind11.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/Exceptions.h>

#include <torch/csrc/autograd/saved_variable.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/anomaly_mode.h>

#include <ATen/Tensor.h>
#include <torch/csrc/THP.h>

#include <cstdint>
#include <list>
#include <memory>
#include <sstream>

namespace torch { namespace autograd {

void throw_python_error() {
  python_error err;
  err.persist();
  throw err;
}

PyObject *THPVariableClass = nullptr;

static PyObject* THPVariable_NewWithVar_actnn(PyTypeObject* type, Variable var)
{
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = (THPVariable*) obj;
    new (&v->cdata) Variable(std::move(var));
    torch::autograd::impl::set_pyobj(v->cdata, obj);
  }
  return obj;
}

PyObject * THPVariable_Wrap_actnn(Variable var)
{
  if (!var.defined()) {
    Py_RETURN_NONE;
  }

  if (auto obj = torch::autograd::impl::pyobj(var)) {
    Py_INCREF(obj);
    return obj;
  }

  return THPVariable_NewWithVar_actnn((PyTypeObject *)THPVariableClass, std::move(var));
}

PyObject* actnn_quantize(const Variable& variable) {
    // get module
    pybind11::gil_scoped_acquire gil;
    auto actnn_module = THPObjectPtr(PyImport_ImportModule("actnn.ops"));
    if (!actnn_module) throw_python_error();

    // prepare inputs
    int num_inputs = 2;
    THPObjectPtr pyInputs(PyTuple_New(num_inputs));
    PyObject* py_tensor = THPVariable_Wrap_actnn(variable);
    if (!py_tensor) throw_python_error();
    PyTuple_SET_ITEM(pyInputs.get(), 0, py_tensor);
    Py_INCREF(Py_None);
    PyTuple_SET_ITEM(pyInputs.get(), 1, Py_None);

    // quantize
    THPObjectPtr quantize_fn(PyObject_GetAttrString(actnn_module, "quantize_activation"));
    if (!quantize_fn) throw_python_error();
    THPObjectPtr r(PyObject_CallObject(quantize_fn, pyInputs.get()));
    if (!r)  throw_python_error();
    ensure_tuple(r);

    return r.release();
}

Variable actnn_dequantize(PyObject* quantized, PyObject* input_shape) {
    // get module
    pybind11::gil_scoped_acquire gil;
    auto actnn_module = THPObjectPtr(PyImport_ImportModule("actnn.ops"));
    if (!actnn_module) throw_python_error();

    // prepare inputs
    int num_inputs = 2;
    THPObjectPtr pyInputs(PyTuple_New(num_inputs));
    PyTuple_SET_ITEM(pyInputs.get(), 0, quantized);
    PyTuple_SET_ITEM(pyInputs.get(), 1, input_shape);

    // dequantize
    THPObjectPtr dequantize_fn(PyObject_GetAttrString(actnn_module, "dequantize_activation"));
    if (!dequantize_fn) throw_python_error();
    THPObjectPtr r(PyObject_CallObject(dequantize_fn, pyInputs.get()));
    if (!r) throw_python_error();
    ensure_tuple(r);

    // extract outputs
    int num_outputs = PyTuple_GET_SIZE(r.get());
    TORCH_CHECK(num_outputs == 1, "Get ",  num_outputs, " outputs, expect 1");
    THPVariable* q_var = (THPVariable*) PyTuple_GET_ITEM(r.get(), 0);
    return q_var->cdata;
}

SavedVariable::SavedVariable(const Variable& variable, bool is_output, bool is_inplace_view) {
  if (variable.defined()) {
    was_default_constructed_ = false;
    output_nr_ = variable.output_nr();
    requires_grad_ = variable.requires_grad();
    has_grad_fn_ = !variable.is_leaf();
    is_inplace_view_ = is_inplace_view;
    // These copies are all shared_ptr copies, so slightly more expensive.
    // Do them here instead of in the init list in case data is undefined.
    // data_ = variable.tensor_data();
    if (has_grad_fn_) {
      is_quantized_ = true;
      quantized_ = actnn_quantize(variable);
      input_sizes_ = PyTuple_New(variable.dim());
      for (int i = 0; i <variable.dim(); i++) {
          PyTuple_SET_ITEM(input_sizes_, i, PyLong_FromLongLong(variable.sizes().data()[i]));
      }
    } else {
      data_ = variable.tensor_data();
    }
    if (variable.is_leaf()) {
      grad_accumulator_ = impl::grad_accumulator(variable);
    } else if (!is_output) {
      grad_fn_ = variable.grad_fn();
    } else if (is_inplace_view) {
      weak_grad_fn_ = variable.grad_fn();
    }
    version_counter_ = impl::version_counter(variable);
    saved_version_ = version_counter_.current_version();
  }
}

SavedVariable::SavedVariable(const c10::optional<Variable>& variable, bool is_output, bool is_inplace_view)
  : SavedVariable(variable.has_value() ? *variable : Variable(), is_output, is_inplace_view) {}

Variable SavedVariable::unpack(std::shared_ptr<Node> saved_for) const {
  if (quantized_ == NULL && !data_.defined()) {
    if (!was_default_constructed_) {
      throw std::runtime_error(ERR_BACKWARD_TWICE);
    }
    return Variable();
  }

  Variable quantized_data;
  if (!is_quantized_) {
    quantized_data = data_;
  } else {
    quantized_data = actnn_dequantize(quantized_, input_sizes_);
  }

  // if (quantized_data.dim() > 0)
  //   std::cout <<"----Saved variable shape----" << quantized_data.sizes() << quantized_data.data()[0] << "\n";

  auto grad_fn = is_inplace_view_ ? weak_grad_fn_.lock() : grad_fn_;
  if (has_grad_fn_ && !grad_fn) {
    if (!saved_for) {
      // If saving the grad_fn would create a circular reference, then it must
      // be passed in to the unpack function.
      throw std::runtime_error("No grad_fn for non-leaf saved variable");
    }
    grad_fn = std::move(saved_for);
  }

  if (saved_version_ != version_counter_.current_version()) {
    std::stringstream message;
    message << "one of the variables needed for gradient computation has been "
        // "modified by an inplace operation: [" << data_.toString() << " "
        // << data_.sizes() << "]";
        "modified by an inplace operation: [" << quantized_data.toString() << " "
        << quantized_data.sizes() << "]";
    if (grad_fn) {
        message << ", which is output " << output_nr_
            << " of " << grad_fn->name() << ",";
    }
    message << " is at version " << version_counter_.current_version()
        << "; expected version " << saved_version_ << " instead.";
    if (!AnomalyMode::is_enabled()) {
        message << " Hint: enable anomaly detection to find the operation "
            "that failed to compute its gradient, with torch.autograd."
            "set_detect_anomaly(True).";
    }
    else {
        message << " Hint: the backtrace further above shows the operation "
            "that failed to compute its gradient. The variable in question "
            "was changed in there or anywhere later. Good luck!";
    }
    throw std::runtime_error(message.str());
  }

  // NB: saved views are unpacked as normal Variables (not views) even though
  // they still share the same storage. This works only because we never call
  // in-place functions on unpacked variables.
  Variable var;
  if (grad_fn) {
    // var = make_variable(data_, Edge(std::move(grad_fn), output_nr_));
    var = make_variable(quantized_data, Edge(std::move(grad_fn), output_nr_));
  } else {
    // var = make_variable(data_, requires_grad_);
    var = make_variable(quantized_data, requires_grad_);
  }
  impl::set_version_counter(var, saved_version_);

  // If a Variable is a leaf (no grad_fn saved), and it requires_grad, then we
  // should have saved the grad accumulator. Even if the Variable no longer
  // alive, the accumulator should be kept alive by the references in the
  // graph).
  if (requires_grad_ && !var.grad_fn() && grad_accumulator_.expired())
    throw std::logic_error("No grad accumulator for a saved leaf!");
  impl::set_grad_accumulator(var, grad_accumulator_);
  return var;
}

const char* ERR_BACKWARD_TWICE =
    "Trying to backward through the graph a second time, but the saved intermediate "
    "results have already been freed. Specify retain_graph=True when calling "
    "backward the first time.";

}} // namespace torch::autograd
