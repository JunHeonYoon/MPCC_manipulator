#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <eigenpy/eigenpy.hpp>
#include <Eigen/Dense>

#include "config.h"
#include "types.h"
#include "Params/params.h"
#include "Model/robot_model.h"
#include "Constraints/SelfCollision/SelfCollisionModel.h"
// #include "Constraints/EnvCollision/EnvCollisionModel.h"
#include "Model/integrator.h"
#include "Params/track.h"
#include "Spline/cubic_spline_rot.h"
#include "Spline/arc_length_spline.h"
#include "MPC/mpc.h"

using namespace boost::python;
using namespace mpcc;

// Converter for std::vector<Eigen::Matrix3d>
struct VectorEigenMatrix3d_to_python
{
    static PyObject* convert(const std::vector<Eigen::Matrix3d>& vec)
    {
        boost::python::list py_list;
        for (const auto& mat : vec)
        {
            py_list.append(mat);
        }
        return incref(py_list.ptr());
    }
};

struct VectorEigenMatrix3d_from_python
{
    VectorEigenMatrix3d_from_python()
    {
        converter::registry::push_back(&convertible, &construct, boost::python::type_id<std::vector<Eigen::Matrix3d>>());
    }

    static void* convertible(PyObject* obj_ptr)
    {
        if (!PySequence_Check(obj_ptr)) return nullptr;
        return obj_ptr;
    }

    static void construct(PyObject* obj_ptr, converter::rvalue_from_python_stage1_data* data)
    {
        void* storage = ((converter::rvalue_from_python_storage<std::vector<Eigen::Matrix3d>>*)data)->storage.bytes;
        new (storage) std::vector<Eigen::Matrix3d>();
        std::vector<Eigen::Matrix3d>& vec = *(std::vector<Eigen::Matrix3d>*)(storage);

        int len = PySequence_Size(obj_ptr);
        if (len < 0) throw_error_already_set();
        vec.reserve(len);

        for (int i = 0; i < len; ++i)
        {
            vec.push_back(extract<Eigen::Matrix3d>(PySequence_GetItem(obj_ptr, i)));
        }

        data->convertible = storage;
    }
};

struct PairConverter {
    static PyObject* convert(const std::pair<Eigen::VectorXd, Eigen::MatrixXd>& p) {
        boost::python::tuple t = boost::python::make_tuple(p.first, p.second);
        return boost::python::incref(t.ptr());
    }
};

// Converter: std::vector<float> to numpy array
struct VectorFloat_to_numpy
{
    static PyObject* convert(const std::vector<float>& vec)
    {
        boost::python::list py_list;
        for (const auto& mat : vec)
        {
            py_list.append(mat);
        }
        return incref(py_list.ptr());
    }
};

// Converter: numpy array to std::vector<float>
struct Numpy_to_VectorFloat
{
    Numpy_to_VectorFloat()
    {
        boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<std::vector<float>>());
    }

    static void* convertible(PyObject* obj_ptr)
    {
        return PyArray_Check(obj_ptr) ? obj_ptr : nullptr;
    }

    static void construct(PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data)
    {
        PyArrayObject* array = reinterpret_cast<PyArrayObject*>(obj_ptr);
        void* storage = ((boost::python::converter::rvalue_from_python_storage<std::vector<float>>*)data)->storage.bytes;

        npy_intp size = PyArray_SIZE(array);
        float* raw_data = reinterpret_cast<float*>(PyArray_DATA(array));

        new (storage) std::vector<float>(raw_data, raw_data + size);
        data->convertible = storage;
    }
};


BOOST_PYTHON_MODULE(MPCC_WRAPPER)
{
    eigenpy::enableEigenPy();
    to_python_converter<std::vector<Eigen::Matrix3d>, VectorEigenMatrix3d_to_python>();
    VectorEigenMatrix3d_from_python();
    to_python_converter<std::pair<Eigen::VectorXd, Eigen::MatrixXd>, PairConverter>();
    to_python_converter<std::vector<float>, VectorFloat_to_numpy>();
    Numpy_to_VectorFloat();

    // =================================================
    // =================== config.h ====================
    // =================================================
    // Binding constants
    scope().attr("PANDA_DOF") = PANDA_DOF;
    scope().attr("NX") = NX;
    scope().attr("NU") = NU;
    scope().attr("NPC") = NPC;
    scope().attr("N") = N;
    scope().attr("INF") = INF;
    scope().attr("N_SPLINE") = N_SPLINE;

    // Binding StateInputIndex
    class_<StateInputIndex>("StateInputIndex")
        .def_readonly("q1", &StateInputIndex::q1)
        .def_readonly("q2", &StateInputIndex::q2)
        .def_readonly("q3", &StateInputIndex::q3)
        .def_readonly("q4", &StateInputIndex::q4)
        .def_readonly("q5", &StateInputIndex::q5)
        .def_readonly("q6", &StateInputIndex::q6)
        .def_readonly("q7", &StateInputIndex::q7)
        .def_readonly("s", &StateInputIndex::s)
        .def_readonly("vs", &StateInputIndex::vs)
        .def_readonly("dq1", &StateInputIndex::dq1)
        .def_readonly("dq2", &StateInputIndex::dq2)
        .def_readonly("dq3", &StateInputIndex::dq3)
        .def_readonly("dq4", &StateInputIndex::dq4)
        .def_readonly("dq5", &StateInputIndex::dq5)
        .def_readonly("dq6", &StateInputIndex::dq6)
        .def_readonly("dq7", &StateInputIndex::dq7)
        .def_readonly("dVs", &StateInputIndex::dVs)
        .def_readonly("con_selcol", &StateInputIndex::con_selcol)
        .def_readonly("con_sing", &StateInputIndex::con_sing)
        .def_readonly("con_envcol", &StateInputIndex::con_envcol)
    ;

    // Bind the static instance si_index
    scope().attr("si_index") = si_index;

    // Binding pkg_path
    scope().attr("pkg_path") = pkg_path;

    // // =================================================
    // // =================== types.h ====================
    // // =================================================
    class_<State>("State")
        .def_readwrite("q1", &State::q1)
        .def_readwrite("q2", &State::q2)
        .def_readwrite("q3", &State::q3)
        .def_readwrite("q4", &State::q4)
        .def_readwrite("q5", &State::q5)
        .def_readwrite("q6", &State::q6)
        .def_readwrite("q7", &State::q7)
        .def_readwrite("s", &State::s)
        .def_readwrite("vs", &State::vs)
        .def("setZero", &State::setZero)
        .def("unwrap", &State::unwrap)
    ;

    class_<Input>("Input")
        .def_readwrite("dq1", &Input::dq1)
        .def_readwrite("dq2", &Input::dq2)
        .def_readwrite("dq3", &Input::dq3)
        .def_readwrite("dq4", &Input::dq4)
        .def_readwrite("dq5", &Input::dq5)
        .def_readwrite("dq6", &Input::dq6)
        .def_readwrite("dq7", &Input::dq7)
        .def_readwrite("dVs", &Input::dVs)
        .def("setZero", &Input::setZero)
    ;

    class_<PathToJson>("PathToJson")
        .def_readwrite("param_path", &PathToJson::param_path)
        .def_readwrite("cost_path", &PathToJson::cost_path)
        .def_readwrite("bounds_path", &PathToJson::bounds_path)
        .def_readwrite("track_path", &PathToJson::track_path)
        .def_readwrite("normalization_path", &PathToJson::normalization_path)
        .def_readwrite("sqp_path", &PathToJson::sqp_path)
    ;

    class_<std::map<std::string, double>>("StringDoubleMap")
        .def(map_indexing_suite<std::map<std::string, double>>());

    class_<ParamValue>("ParamValue")
        .def_readwrite("param", &ParamValue::param)
        .def_readwrite("cost", &ParamValue::cost)
        .def_readwrite("bounds", &ParamValue::bounds)
        .def_readwrite("normalization", &ParamValue::normalization)
        .def_readwrite("sqp", &ParamValue::sqp)
    ;

    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, NX, 1>>();        // For StateVector
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, PANDA_DOF, 1>>(); // For JointVector, dJointVector
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, NU, 1>>();        // For InputVector
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, NX, NX>>();       // For A_MPC, Q_MPC, TX_MPC
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, NX, NU>>();       // For B_MPC, S_MPC
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, NX, 1>>();        // For g_MPC, q_MPC, Bounds_x
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, NU, NU>>();       // For R_MPC, TU_MPC
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, NU, 1>>();        // For r_MPC, Bounds_u
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, NPC, NX>>();      // For C_MPC
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, 1, NX>>();        // For C_i_MPC
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, NPC, NU>>();      // For D_MPC
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, 1, NU>>();        // For D_i_MPC
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, NPC, 1>>();       // For d_MPC

    // Expose conversion functions
    def("stateToVector", stateToVector);
    def("stateToJointVector", stateToJointVector);
    def("inputTodJointVector", inputTodJointVector);
    def("inputToVector", inputToVector);
    def("vectorToState", vectorToState);
    def("vectorToInput", vectorToInput);
    def("arrayToState", arrayToState);
    def("arrayToInput", arrayToInput);

    // =================================================
    // ================ robot_model.h ==================
    // =================================================
    class_<RobotModel, std::shared_ptr<RobotModel>, boost::noncopyable>("RobotModel", init<>())
        .def("getNumq", &RobotModel::getNumq, return_value_policy<copy_const_reference>())
        .def("getNumv", &RobotModel::getNumv, return_value_policy<copy_const_reference>())
        .def("getNumu", &RobotModel::getNumu, return_value_policy<copy_const_reference>())
        .def("getUpdateKinematics", &RobotModel::getUpdateKinematics)
        .def("getJacobian", static_cast<const MatrixXd& (RobotModel::*)(const int&)>(&RobotModel::getJacobian), return_value_policy<copy_const_reference>())
        .def("getJacobian", static_cast<const MatrixXd& (RobotModel::*)(const VectorXd&)>(&RobotModel::getJacobian), return_value_policy<copy_const_reference>())
        .def("getJacobian", static_cast<const MatrixXd& (RobotModel::*)(const VectorXd&, const int&)>(&RobotModel::getJacobian), return_value_policy<copy_const_reference>())
        .def("getJacobianv", static_cast<const MatrixXd& (RobotModel::*)(const VectorXd&, const int&)>(&RobotModel::getJacobianv), return_value_policy<copy_const_reference>())
        .def("getJacobianv", static_cast<const MatrixXd& (RobotModel::*)(const VectorXd&)>(&RobotModel::getJacobianv), return_value_policy<copy_const_reference>())
        .def("getJacobianw", static_cast<const MatrixXd& (RobotModel::*)(const VectorXd&, const int&)>(&RobotModel::getJacobianw), return_value_policy<copy_const_reference>())
        .def("getJacobianw", static_cast<const MatrixXd& (RobotModel::*)(const VectorXd&)>(&RobotModel::getJacobianw), return_value_policy<copy_const_reference>())
        .def("getPosition", static_cast<const Vector3d& (RobotModel::*)(const int&)>(&RobotModel::getPosition), return_value_policy<copy_const_reference>())
        .def("getEEPosition", static_cast<const Vector3d& (RobotModel::*)()>(&RobotModel::getEEPosition), return_value_policy<copy_const_reference>())
        .def("getEEPosition", static_cast<const Vector3d& (RobotModel::*)(const VectorXd&)>(&RobotModel::getEEPosition), return_value_policy<copy_const_reference>())
        .def("getOrientation", static_cast<const Matrix3d& (RobotModel::*)(const int&)>(&RobotModel::getOrientation), return_value_policy<copy_const_reference>())
        .def("getEEOrientation", static_cast<const Matrix3d& (RobotModel::*)()>(&RobotModel::getEEOrientation), return_value_policy<copy_const_reference>())
        .def("getEEOrientation", static_cast<const Matrix3d& (RobotModel::*)(const VectorXd&)>(&RobotModel::getEEOrientation), return_value_policy<copy_const_reference>())
        .def("getTransformation", static_cast<const Affine3d& (RobotModel::*)(const int&)>(&RobotModel::getTransformation), return_value_policy<copy_const_reference>())
        .def("getEETransformation", static_cast<const Affine3d& (RobotModel::*)()>(&RobotModel::getEETransformation), return_value_policy<copy_const_reference>())
        .def("getEETransformation", static_cast<const Affine3d& (RobotModel::*)(const VectorXd&)>(&RobotModel::getEETransformation), return_value_policy<copy_const_reference>())
        .def("getJointPosition", &RobotModel::getJointPosition, return_value_policy<copy_const_reference>())
        .def("getMassMatrix", &RobotModel::getMassMatrix, return_value_policy<copy_const_reference>())
        .def("getNonlinearEffect", &RobotModel::getNonlinearEffect, return_value_policy<copy_const_reference>())
        .def("getManipulability", static_cast<const double& (RobotModel::*)(const VectorXd&)>(&RobotModel::getManipulability), return_value_policy<copy_const_reference>())
        .def("getManipulability", static_cast<const double& (RobotModel::*)(const VectorXd&, const int&)>(&RobotModel::getManipulability), return_value_policy<copy_const_reference>())
        .def("getDManipulability", static_cast<const VectorXd& (RobotModel::*)(const VectorXd&)>(&RobotModel::getDManipulability), return_value_policy<copy_const_reference>())
        .def("getDManipulability", static_cast<const VectorXd& (RobotModel::*)(const VectorXd&, const int&)>(&RobotModel::getDManipulability), return_value_policy<copy_const_reference>())
    ;

    // =================================================
    // ============= SelfCollisionModel.h ==============
    // =================================================
    // SelCollNNmodel binding
    class_<SelCollNNmodel, std::shared_ptr<SelCollNNmodel>, boost::noncopyable>("SelCollNNmodel", no_init)
            .def(init<>())
            .def(init<const std::string&>())
            .def("setNeuralNetwork", &SelCollNNmodel::setNeuralNetwork)
            .def("calculateMlpOutput", &SelCollNNmodel::calculateMlpOutput)
    ;

    // =================================================
    // ============== EnvCollisionModel.h ==============
    // =================================================
    // EnvCollNNmodel binding
    // class_<EnvCollNNmodel, std::shared_ptr<EnvCollNNmodel>, boost::noncopyable>("EnvCollNNmodel", no_init)
    //         .def(init<>())
    //         .def(init<const std::string&>())
    //         .def("setNeuralNetwork", &EnvCollNNmodel::setNeuralNetwork)
    //         .def("forward", &EnvCollNNmodel::forward)
    // ;
    
    // =================================================
    // ================= integrator.h ==================
    // =================================================
    // Integrator binding
    class_<Integrator>("Integrator", init<>())
        .def(init<double, const PathToJson &>())
        .def("RK4", &Integrator::RK4)
        .def("EF", &Integrator::EF)
        .def("simTimeStep", &Integrator::simTimeStep)
    ;

    // =================================================
    // =================== track.h =====================
    // =================================================
    // Track binding
    class_<Track>("Track", init<std::string>())
        .def("getTrack", &Track::getTrack)
    ;

    // TrackPos binding
    class_<TrackPos>("TrackPos", no_init)
        .add_property("X", make_getter(&TrackPos::X, return_value_policy<copy_const_reference>()))
        .add_property("Y", make_getter(&TrackPos::Y, return_value_policy<copy_const_reference>()))
        .add_property("Z", make_getter(&TrackPos::Z, return_value_policy<copy_const_reference>()))
        .add_property("R", make_getter(&TrackPos::R, return_value_policy<copy_const_reference>()))
    ;

    // =================================================
    // ============== cubic_spline_rot.h ===============
    // =================================================
    // Bindings for utility functions
    def("getSkewMatrix", &getSkewMatrix);
    def("getInverseSkewVector", &getInverseSkewVector);
    def("LogMatrix", &LogMatrix);
    def("ExpMatrix", &ExpMatrix);

    // =================================================
    // ============== arc_length_spline.h ==============
    // =================================================
    // PathData binding
    class_<PathData>("PathData")
        .def_readwrite("X", &PathData::X)
        .def_readwrite("Y", &PathData::Y)
        .def_readwrite("Z", &PathData::Z)
        .add_property("R", make_getter(&PathData::R, return_value_policy<copy_non_const_reference>()))
        // .def_readwrite("R", &PathData::R)
        .def_readwrite("s", &PathData::s)
        .def_readwrite("n_points", &PathData::n_points)
    ;

    // ArcLengthSpline binding
    class_<ArcLengthSpline>("ArcLengthSpline", init<>())
        .def(init<const PathToJson &>())
        .def(init<const PathToJson &, const ParamValue &>())
        .def("gen6DSpline", &ArcLengthSpline::gen6DSpline)
        .def("getPosition", &ArcLengthSpline::getPosition)
        .def("getOrientation", &ArcLengthSpline::getOrientation)
        .def("getDerivative", &ArcLengthSpline::getDerivative)
        .def("getOrientationDerivative", &ArcLengthSpline::getOrientationDerivative)
        .def("getSecondDerivative", &ArcLengthSpline::getSecondDerivative)
        .def("getLength", &ArcLengthSpline::getLength)
        .def("projectOnSpline", &ArcLengthSpline::projectOnSpline)
        .def("getPathData", &ArcLengthSpline::getPathData)
    ;

    // =================================================
    // ===================== mpc.h =====================
    // =================================================
    // MPC binding
    class_<MPC, std::shared_ptr<MPC>, boost::noncopyable>("MPC", init<>())
        .def(init<double, const PathToJson &>())
        .def(init<double, const PathToJson &, const ParamValue &>())
        .def("runMPC", &MPC::runMPC)
        .def("runMPC_", &MPC::runMPC_)
        .def("setTrack", &MPC::setTrack)
        .def("getTrackLength", &MPC::getTrackLength)
        .def("setParam", &MPC::setParam)
    ;

    class_<ComputeTime>("ComputeTime")
        .def_readwrite("set_qp", &ComputeTime::set_qp)
        .def_readwrite("solve_qp", &ComputeTime::solve_qp)
        .def_readwrite("get_alpha", &ComputeTime::get_alpha)
        .def_readwrite("set_env", &ComputeTime::set_env)
        .def_readwrite("total", &ComputeTime::total)
        .def("setZero", &ComputeTime::setZero)
    ;

    class_<OptVariables>("OptVariables")
        .def_readwrite("xk", &OptVariables::xk)
        .def_readwrite("uk", &OptVariables::uk)
    ;

    class_<std::vector<OptVariables>>("std::vector<OptVariables>")
        .def(vector_indexing_suite<std::vector<OptVariables>>());

    // MPCReturn binding
    class_<MPCReturn>("MPCReturn")
        .def_readwrite("u0", &MPCReturn::u0)
        .def_readwrite("mpc_horizon", &MPCReturn::mpc_horizon)
        .def_readwrite("compute_time", &MPCReturn::compute_time)
        .def("setZero", &MPCReturn::setZero)
    ;
}