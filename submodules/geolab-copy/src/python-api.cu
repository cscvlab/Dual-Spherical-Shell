#include<sdfgenerator.cuh>
#include<renderer/renderer.cuh>
// #include<renderer/glrenderer.cuh>
#include<sampler/sphere_sampler.h>
#include<sampler/triangle_sampler.cuh>
#include<geometry/voxel.cuh>
#include<ray_tracer_utils.cuh>


#include<pybind11/pybind11.h>
#include<pybind11/chrono.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>
#include<pybind11/eigen.h>
#include<pybind11/functional.h>

namespace py = pybind11;

PYBIND11_MODULE(pygeo, m){
    // Load obj

    py::enum_<SDFCalcMode>(m, "SDFCalcMode")
        .value("WATERTIGHT", SDFCalcMode::WATERTIGHT)
        .value("RAYSTAB", SDFCalcMode::RAYSTAB)
        .value("PATHESCAPE", SDFCalcMode::PATHESCAPE);

    py::class_<Mesh::Vertex>(m, "Vertex")
        .def_readwrite("position", &Mesh::Vertex::position)
        .def_readwrite("normal", &Mesh::Vertex::normal)
        .def_readwrite("textureCoord", &Mesh::Vertex::textureCoord);

    py::class_<Mesh>(m, "Mesh")
        .def(py::init<>())
        .def("load_triangles", py::overload_cast<std::string>(&Mesh::load_triangles));

    py::class_<Triangle>(m, "Triangle")
        .def_readwrite("a", &Triangle::a)
        .def_readwrite("b", &Triangle::b)
        .def_readwrite("c", &Triangle::c);

    py::class_<BoundingBox>(m, "BoundingBox")
        .def(py::init<>())
        .def_readwrite("min", &BoundingBox::min)
        .def_readwrite("max", &BoundingBox::max);

    py::class_<SDFGenerator>(m, "SDFGenerator")
        .def(py::init<>())
        .def("load_obj", &SDFGenerator::load_obj)
        .def("generate_sdf", &SDFGenerator::generate_sdf, "generate sdf", 
            py::arg("positions"),
            py::arg("mode")    
        );

    py::class_<SDFPayload>(m, "SDFPayload")
        .def_readwrite("dir", &SDFPayload::dir)
        .def_readwrite("idx", &SDFPayload::idx)
        .def_readwrite("n_steps", &SDFPayload::n_steps)
        .def_readwrite("alive", &SDFPayload::alive);

    
    m.def("sample_fibonacci", py::overload_cast<unsigned int>(&sampler_fibonacci));
    m.def("sample_voxel_center", py::overload_cast<Eigen::Vector3i>(&sample_voxel_center));

    m.def("sample_on_triangles", py::overload_cast<std::vector<Triangle>&, size_t, size_t, size_t>(&sample_on_triangles));
    m.def("sample_on_triangles", py::overload_cast<std::vector<Triangle>&, size_t, Eigen::Vector3f>(&sample_on_triangles));

    py::enum_<ERenderMode>(m, "ERenderMode")
        .value("LATTICE", ERenderMode::LATTICE)
        .value("NORMONLY", ERenderMode::NORMONLY)
        .value("INSTANT_AO", ERenderMode::INSTANT_AO)
        .value("POSITION", ERenderMode::POSITION)
        .value("DEPTH", ERenderMode::DEPTH)
        .value("COST", ERenderMode::COST)
        .value("DISTANCE", ERenderMode::DISTANCE);

    py::class_<BRDFParams>(m, "BRDFParams")
        .def_readwrite("metallic", &BRDFParams::metallic)
        .def_readwrite("subsurface", &BRDFParams::subsurface)
        .def_readwrite("roughness", &BRDFParams::roughness)
        .def_readwrite("sheen", &BRDFParams::sheen)
        .def_readwrite("clearcoat", &BRDFParams::clearcoat)
        .def_readwrite("clearcoat_gloss", &BRDFParams::clearcoat_gloss);

    py::class_<Light>(m, "Light")
        .def_readwrite("pos", &Light::pos)
        .def_readwrite("light_color", &Light::light_color)
        .def_readwrite("ambient_color", &Light::ambient_color)
        .def_readwrite("background_color", &Light::background_color)
        .def_readwrite("specular", &Light::specular)
        .def_readwrite("kd", &Light::kd);
        
    py::class_<Scene>(m, "Scene")
        .def_readwrite("surfaceColor", &Scene::surface_color)
        .def_readwrite("slice_plane_z", &Scene::slice_plane_z)
        .def_readwrite("floor_y", &Scene::floor_y)
        .def_readwrite("aabb", &Scene::aabb)
        .def_readwrite("aabb_offset", &Scene::aabb_offset);

    py::class_<SDFRenderer>(m, "SDFRenderer")
        .def(py::init<Eigen::Vector2i, bool>(), 
            py::arg("win_res") = Eigen::Vector2i(1920, 1080),
            py::arg("gui") = true)
        .def("build_bvh", &SDFRenderer::build_bvh)
        .def("render_ray_trace", &SDFRenderer::render_ray_trace, 
            py::arg("cam_pos") = std::vector<Eigen::Vector3f>(),
            py::arg("cam_focus") = Eigen::Vector3f::Zero(),
            py::arg("render_mode") = ERenderMode::LATTICE)
        .def("read_and_render_frame", &SDFRenderer::read_and_render_frame, 
            py::arg("points"),
            py::arg("normals"),
            py::arg("hit"),
            py::arg("n_steps"),
            py::arg("distances"),
            py::arg("pos"),
            py::arg("to") = Eigen::Vector3f::Zero())
        .def_readwrite("light", &SDFRenderer::light)
        .def_readwrite("scene", &SDFRenderer::scene)
        .def_readwrite("render_ground_truth", &SDFRenderer::m_render_ground_truth)
        .def_readwrite("draw_coordinate_axis", &SDFRenderer::draw_coordinate_axis)
        .def_readwrite("draw_light_sphere", &SDFRenderer::draw_light_sphere);

    py::class_<DualVoxel>(m, "DualVoxel")
        .def(py::init<Eigen::Vector3i>())
        .def("surface_num", &DualVoxel::surface_num)
        .def("inside_num", &DualVoxel::inside_num)
        .def("voxel_num", &DualVoxel::voxel_num)
        .def("total_num", &DualVoxel::total_num)
        .def("outside_num", &DualVoxel::outside_num)
        .def("voxel_size", &DualVoxel::voxel_size)
        .def("half_voxel_size", &DualVoxel::half_voxel_size)
        .def("center", &DualVoxel::center)
        .def("point_in_voxel", &DualVoxel::point_in_voxel)
        .def("hash", &DualVoxel::hash)
        .def("voxelize_triangles", &DualVoxel::voxelize_triangles)
        .def("fill_inside", &DualVoxel::fill_inside)
        // .def("save_grid", &DualVoxel::save_grid)
        // .def("load_grid", &DualVoxel::load_grid)
        .def("surface_voxel", &DualVoxel::surface_voxel)
        .def("inside_voxel", &DualVoxel::inside_voxel)
        .def("outside_voxel", &DualVoxel::outside_voxel)
        .def("surface_voxel_center", &DualVoxel::surface_voxel_center)
        .def("inside_voxel_center", &DualVoxel::inside_voxel_center)
        .def("outside_voxel_center", &DualVoxel::outside_voxel_center)
        .def("tags", &DualVoxel::tags)
        .def("voxels", &DualVoxel::voxels)
        .def("voxel_centers", &DualVoxel::voxel_centers);

    py::class_<DualVoxel::HYD>(m, "DualVoxelHYD")
        .def("edge_voxel", &DualVoxel::HYD::edge_voxel)
        .def("subvoxel_centers", &DualVoxel::HYD::subvoxel_centers);

    py::class_<DualVoxelDynamic>(m, "DualVoxelDynamic")
        .def(py::init<SDFCalcMode>())
        .def("load_model", py::overload_cast<std::string>(&DualVoxelDynamic::load_model))
        .def("load_model", py::overload_cast<std::vector<Triangle>&>(&DualVoxelDynamic::load_model))
        .def("surface_voxel", &DualVoxelDynamic::surface_voxel)
        .def("inside_voxel", &DualVoxelDynamic::inside_voxel)
        .def("outside_voxel", &DualVoxelDynamic::outside_voxel)
        .def("surface_voxel_center", &DualVoxelDynamic::surface_voxel_center)
        .def("inside_voxel_center", &DualVoxelDynamic::inside_voxel_center)
        .def("outside_voxel_center", &DualVoxelDynamic::outside_voxel_center)
        .def("judge_voxel", &DualVoxelDynamic::judge_voxel)
        .def("judge_voxel_center", &DualVoxelDynamic::judge_voxel_center);

    py::class_<RayTracerUtils>(m, "RayTracerUtils")
        .def(py::init<>())
        .def(py::init<std::string>())
        .def(py::init<std::vector<Triangle>&>())
        .def("load_mesh", py::overload_cast<std::string>(&RayTracerUtils::load_mesh))
        .def("load_mesh", py::overload_cast<std::vector<Triangle>&>(&RayTracerUtils::load_mesh))
        .def("signed_distance", &RayTracerUtils::signed_distance)
        .def("trace", &RayTracerUtils::trace);


}