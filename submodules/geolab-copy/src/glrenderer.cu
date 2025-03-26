#include<renderer/gl_renderer.cuh>

bool PolygonRenderer::init_window(Eigen::Vector2i win_res){
    // initialize window
    if(!glfwInit()){
        std::cout << "Error: Can not initialize glfw! " << std::endl;
        return 0;
    }
    m_window = glfwCreateWindow(win_res.x(), win_res.y(), "geolab", NULL, NULL);
    if(!m_window){
        glfwTerminate();
        std::cout << "Error: Can not create window! " << std::endl;
        return 0;
    }
  
    glfwMakeContextCurrent(m_window);
  
    //should be done after glfw initialized
    if(glewInit() != GLEW_OK){
        std::cout << "Error: Can not initialize glew" << std::endl;
        return 0;
    }
    std::cout << glGetString(GL_VERSION) << std::endl;

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    io.ConfigInputTrickleEventQueue = false;
    // Setup Platform/Renderer bindings
    if(!ImGui_ImplGlfw_InitForOpenGL(m_window, true))return 0;
    if(!ImGui_ImplOpenGL3_Init("#version 330 core"))return 0;
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    return true;
}

void PolygonRenderer::mouse_drag(ImVec2 rel, int button){
    bool is_left_held = (button & 1) != 0;
    Eigen::Vector3f up = camera.view_up();
    Eigen::Vector3f side = camera.view_side();

	if (is_left_held) {
        Eigen::Vector3f pos = camera.view_pos();
		float alpha = amplitude * rel.x / PI * 180;
        float beta = amplitude * rel.y / PI * 180;
        Eigen::Matrix3f translate =
		(Eigen::AngleAxisf(static_cast<float>(-amplitude * 0.0001 * rel.x / PI * 180), up) * // Scroll sideways around up vector
		Eigen::AngleAxisf(static_cast<float>(-amplitude * 0.0001 * rel.y / PI * 180), side)).matrix(); // Scroll around side vector
        camera.pose = translate * camera.pose;
	}
}

void PolygonRenderer::mouse_scroll(float delta){
    if(delta == 0.0f)return;
    
    if(!ImGui::GetIO().WantCaptureMouse){
        camera.pose.col(3) += delta*amplitude * camera.pose.col(2);
    }
}

bool PolygonRenderer::cursor_event_handler(){
    ImVec2 m = ImGui::GetMousePos();
	int mb = 0;
	float mw = 0.f;
	ImVec2 relm = {};
    if (!ImGui::IsAnyItemActive() && !ImGuizmo::IsUsing() && !ImGuizmo::IsOver()) {
        mw = ImGui::GetIO().MouseWheel;
		relm = ImGui::GetIO().MouseDelta;
		if (ImGui::GetIO().MouseDown[0]) mb |= 1;
		if (ImGui::GetIO().MouseDown[1]) mb |= 2;
		if (ImGui::GetIO().MouseDown[2]) mb |= 4;
		
	}
	mouse_scroll(mw);
	mouse_drag(relm, mb);
    return 1;
}

bool PolygonRenderer::keyboard_event_handler(){
    if (ImGui::GetIO().WantCaptureKeyboard) {
		return false;
	}
    if (ImGui::IsKeyPressed('W')) {
        camera.pose.col(3) += amplitude * camera.pose.col(2);
	}
    if (ImGui::IsKeyPressed('S')) {
        camera.pose.col(3) -= amplitude * camera.pose.col(2);
	}
    if (ImGui::IsKeyPressed('A')) {
        camera.pose.col(3) -= amplitude * camera.pose.col(0);
	}
    if (ImGui::IsKeyPressed('D')) {
        camera.pose.col(3) += amplitude * camera.pose.col(0);
	}
    if (ImGui::IsKeyPressed(ImGuiKey_::ImGuiKey_UpArrow)){
        camera.pose.col(3) -= amplitude * camera.pose.col(1);
    }
    if (ImGui::IsKeyPressed(ImGuiKey_::ImGuiKey_DownArrow)){
        camera.pose.col(3) -= amplitude * camera.pose.col(1);
    }
    return true;
}

void PolygonRenderer::imgui_draw(){
    ImGui::SetNextWindowSize(ImVec2(640, 480));

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::Begin("Setting");

    imgui_general_draw();

    if(ImGui::CollapsingHeader("Camera Setting")){
        imgui_camera_draw();
    }

    if(ImGui::CollapsingHeader("Scene Setting")){
        imgui_scene_draw();
    }

    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void PolygonRenderer::imgui_general_draw(){
    ImGui::Text("elapse: %.3f ms fps: %.3f", elapse, fps);
    ImGui::SliderFloat("amplitude", &amplitude, 0.0f, 1.0f);
    ImGui::Checkbox("draw_coordinate_axis", &draw_coordinate_axis);
}

void PolygonRenderer::imgui_camera_draw(){
    if(ImGui::Button("Focus object")){
        camera.focus();
    }
    ImGui::SameLine();
    if(ImGui::Button("Reset Camera")){
        camera.pose.col(3) = Eigen::Vector3f(0.0f, 0.0f, 4.0f);
        camera.focus();
    }
    ImGui::BeginTable("Camera Pose", 4);
    ImGui::TableNextRow();  // side up gaze pos
    ImGui::TableNextColumn(); ImGui::Text("side");
    ImGui::TableNextColumn(); ImGui::Text("up"); 
    ImGui::TableNextColumn(); ImGui::Text("gaze");
    ImGui::TableNextColumn(); ImGui::Text("pos");
    ImGui::TableNextRow();
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(0,0));
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(0,1));
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(0,2));
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(0,3));
    ImGui::TableNextRow();
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(1,0));
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(1,1));
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(1,2));
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(1,3));
    ImGui::TableNextRow();
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(2,0));
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(2,1));
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(2,2));
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(2,3));
    ImGui::EndTable();
}

void PolygonRenderer::imgui_scene_draw(){
    ImGui::ColorEdit3("Surface Color", &scene.surface_color.x());
    ImGui::ColorEdit3("Sky Color", &scene.sky_color.x());
    ImGui::SliderFloat("slice_plane_z", &scene.slice_plane_z, -10.0f, 2.0f);
    ImGui::SliderFloat("floor_y", &scene.floor_y, -2.0f, 2.0f);
    ImGui::SliderFloat("aabb offset", &scene.aabb_offset, -1.0f, 1.0f);
    ImGui::BeginTable("AABB", 4);
    ImGui::TableNextRow();
    ImGui::TableNextColumn(); ImGui::Text("x");
    ImGui::TableNextColumn(); ImGui::Text("y");
    ImGui::TableNextColumn(); ImGui::Text("z");
    ImGui::TableNextColumn(); ImGui::Text(" ");
    ImGui::TableNextRow();
    ImGui::TableNextColumn(); ImGui::InputFloat("x", &scene.aabb.min.x());
    ImGui::TableNextColumn(); ImGui::InputFloat("y", &scene.aabb.min.y());
    ImGui::TableNextColumn(); ImGui::InputFloat("z", &scene.aabb.min.z());
    ImGui::TableNextColumn(); ImGui::Text("min");
    ImGui::TableNextRow();
    ImGui::TableNextColumn(); ImGui::InputFloat("x", &scene.aabb.max.x());
    ImGui::TableNextColumn(); ImGui::InputFloat("y", &scene.aabb.max.y());
    ImGui::TableNextColumn(); ImGui::InputFloat("z", &scene.aabb.max.z());
    ImGui::TableNextColumn(); ImGui::Text("max");
    ImGui::EndTable();
}


// void PolygonRenderer::draw_gui(){
//     GL_CHECK(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
//     GL_CHECK(glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT));
//     for
// }

bool PolygonRenderer::frame(Drawable &drawable){
    clock_t start, end;
    start = clock();
    Eigen::Vector2f focal_length = camera.calc_focal_length(m_res);

    draw_gui(drawable);
    
    end = clock();
    elapse = (float(end - start)/CLOCKS_PER_SEC)*1000.0f;
    fps = 1000.0f / elapse;

    return !glfwWindowShouldClose(m_window);
}

void PolygonRenderer::draw_gui(Drawable &drawable){
    auto transform = mvp(camera.pose, 0.1f, 50.0f, camera.fov.x(), 1.0f);

    GL_CHECK(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

    drawable.update_uniform(transform);
    drawable.draw();

    imgui_draw();
}

void PolygonRenderer::render(Drawable &polygon){
    init_window(m_res);
    coord = std::unique_ptr<Coordinate>(new Coordinate());
    light_sphere = std::unique_ptr<LightSphere>(new LightSphere());

    while(frame);
}