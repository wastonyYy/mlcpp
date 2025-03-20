#ifndef MLCPP_MAIN_H
#define MLCPP_MAIN_H
#include "utilities.h"
/// C++
#include <algorithm>
#include <signal.h>
#include <string>
#include <sstream>
#include <math.h>
#include <chrono>
/// Eigen, Linear Algebra
#include <Eigen/Eigen> //whole Eigen library
/// OpenCV
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
/// ROS
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <image_transport/image_transport.h>
#include <image_geometry/pinhole_camera_model.h>
// PCL
//defaults
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
//mesh
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/gp3.h>
#include <pcl/PolygonMesh.h>
#include <pcl/filters/uniform_sampling.h>  
#include <pcl/visualization/pcl_visualizer.h>
//conversions
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
//voxel, filters, etc
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
//normal
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>


using namespace std;

// main class
class mlcpp_class{
  public:
    /// basic params
    bool m_cam_init=false, m_pcd_load=false, m_pre_process=false;
    bool m_debug_mode=false;
    // string m_infile;
    string m_meshfile;
    vector<double> m_cam_intrinsic;
    /// MLCPP params
    double m_max_dist = 15.0;
    double m_max_angle = 60.0;
    double m_view_pt_dist = 10.0; //from points
    double m_view_pt_each_dist = 2.0; //between each viewpoints
    double m_view_overlap = 0.1; //overlap bet two viewpoints
    double m_slice_height = 8.0;
    int m_TSP_trial = 100;
    /// MLCPP variables
    image_geometry::PinholeCameraModel m_camera_model;
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> m_normal_estimator;
    pcl::PointXYZ m_pcd_center_point;
    pcl::PointCloud<pcl::PointXYZ> m_cloud_map, m_cloud_center, m_cloud_none_viewed;
    pcl::PointCloud<pcl::PointXYZ> m_cloud_initial_view_point, m_optimized_view_point;
    pcl::PointCloud<pcl::PointNormal> m_cloud_normals;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;      

    pcl::PolygonMesh m_mesh;

    geometry_msgs::PoseArray m_normal_pose_array;
    nav_msgs::Path m_all_layer_path;
    /// ROS
    ros::NodeHandle m_nh;
    ros::Subscriber m_path_calc_sub;
    ros::Publisher m_cloud_map_pub, m_cloud_center_pub, m_cloud_none_viewed_pub;
    ros::Publisher m_initial_view_point_pub, m_optimized_view_point_pub;
    ros::Publisher m_cloud_normal_pub, m_all_layer_path_pub;
    ros::Timer m_visualizing_timer;
    // mesh
    ros::Publisher m_pubCloud;


    /// Functions    
    //ROS
    void calc_cb(const std_msgs::Empty::ConstPtr& msg);
    void visualizer_timer_func(const ros::TimerEvent& event);
    //init
    void cam_init();
    // void load_pcd();
    void preprocess_pcd();
    void load_mesh();
    void preprocess_ply();
    void pointCloudandMeshViewer(const pcl::PolygonMesh& mesh);


    //others
    bool check_cam_in(Eigen::VectorXd view_point_xyzpy,pcl::PointXYZ point,pcl::Normal normal);
    void flip_normal(pcl::PointXYZ base,pcl::PointXYZ center,float & nx,float & ny, float & nz);
    Eigen::Matrix3d RPYtoR(double roll, double pitch, double yaw);
    void TwoOptSwap(pcl::PointCloud<pcl::PointNormal>::Ptr pclarray, int start, int finish);
    double PclArrayCost(pcl::PointCloud<pcl::PointNormal>::Ptr pclarray, double &distance);
    double TwoOptTSP(pcl::PointCloud<pcl::PointNormal>::Ptr pclarray);
    void OrdreringTSP(pcl::PointCloud<pcl::PointNormal>::Ptr pclarray);
    //检查可视点到目标点的连线是否被 mesh 面片阻挡
    bool isIntersecting(const pcl::PointXYZ& A, const pcl::PointXYZ& B, const pcl::PointXYZ& v0, const pcl::PointXYZ& v1, const pcl::PointXYZ& v2);
    bool check_viewpoint_collision(Eigen::VectorXd view_point_xyzpy, pcl::PointXYZ point, pcl::Normal normal);
    void buildKDTree(); 
    
    //constructor
    mlcpp_class(const ros::NodeHandle& n);
    ~mlcpp_class(){};
};


//  can be separated into .cpp file
/// class constructor
mlcpp_class::mlcpp_class(const ros::NodeHandle& n) : m_nh(n){
  // params
  m_nh.param<string>("/meshfile", m_meshfile, "resource/bun_zipper.ply");
  // m_nh.param<string>("/infile", m_infile, "resource/bigben2.pcd");
  m_nh.param<bool>("/debug_mode", m_debug_mode, false);
  m_nh.getParam("/cam_intrinsic", m_cam_intrinsic);
  m_nh.param("/slice_height", m_slice_height, 8.0);
  m_nh.param("/max_dist", m_max_dist, 15.0);
  m_nh.param("/max_angle", m_max_angle, 60.0);
  m_nh.param("/view_pt_dist", m_view_pt_dist, 10.0);
  m_nh.param("/view_pt_each_dist", m_view_pt_each_dist, 2.0);
  m_nh.param("/view_overlap", m_view_overlap, 0.1);
  m_nh.param("/TSP_trial", m_TSP_trial, 100);

  //sub
  m_path_calc_sub = m_nh.subscribe<std_msgs::Empty>("/calculate_cpp", 3, &mlcpp_class::calc_cb, this);

  ///pub
  m_cloud_map_pub = m_nh.advertise<sensor_msgs::PointCloud2>("/pcl_map", 3);
  m_cloud_center_pub = m_nh.advertise<sensor_msgs::PointCloud2>("/pcl_center", 3);
  m_cloud_none_viewed_pub = m_nh.advertise<sensor_msgs::PointCloud2>("/none_viewed_pcl", 3);
  m_initial_view_point_pub = m_nh.advertise<sensor_msgs::PointCloud2>("/initial_viewpoints", 3);
  m_optimized_view_point_pub = m_nh.advertise<sensor_msgs::PointCloud2>("/optimized_viewpoints", 3);
  m_cloud_normal_pub = m_nh.advertise<geometry_msgs::PoseArray>("/pcl_normals", 3);
  m_all_layer_path_pub = m_nh.advertise<nav_msgs::Path>("/mlcpp_path", 3);
  // mesh
  m_pubCloud = m_nh.advertise<sensor_msgs::PointCloud2>("/mesh_cloud", 3);

  ///timer
  m_visualizing_timer = m_nh.createTimer(ros::Duration(1/5.0), &mlcpp_class::visualizer_timer_func, this);

  ///init
  cam_init(); //Set camera parameter
  // load_pcd(); //Get Map from pcd
  // preprocess_pcd(); //Preprocess pcd: ground removal, normal estimation

  load_mesh(); //Get mesh from ply

  // mesh_to_pointcloud();

  preprocess_ply(); //Preprocess ply: normal estimation
}

bool mlcpp_class::isIntersecting(const pcl::PointXYZ& A, const pcl::PointXYZ& B, const pcl::PointXYZ& v0, const pcl::PointXYZ& v1, const pcl::PointXYZ& v2) {
    Eigen::Vector3d dir(B.x - A.x, B.y - A.y, B.z - A.z);
    Eigen::Vector3d edge1(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    Eigen::Vector3d edge2(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    Eigen::Vector3d pvec = dir.cross(edge2);
    double det = edge1.dot(pvec);

    if (fabs(det) < 1e-8) return false; // 线平行于三角形
    double invDet = 1.0 / det;

    Eigen::Vector3d tvec(A.x - v0.x, A.y - v0.y, A.z - v0.z);
    double u = tvec.dot(pvec) * invDet;
    if (u < 0 || u > 1) return false;

    Eigen::Vector3d qvec = tvec.cross(edge1);
    double v = dir.dot(qvec) * invDet;
    if (v < 0 || u + v > 1) return false;

    double t = edge2.dot(qvec) * invDet;
        return (t > 1e-8 && t < 1.0 + 1e-8); // t 表示交点在线段 AB 之间
}

///// functions
void mlcpp_class::cam_init(){
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  //Set camera parameter
  sensor_msgs::CameraInfo cam_info;
  cam_info.width = (int)m_cam_intrinsic[0];
  cam_info.height = (int)m_cam_intrinsic[1];
  cam_info.distortion_model = "plumb_bob";
  boost::array<double,9> array_K;
  array_K[0] = m_cam_intrinsic[2];array_K[1] = 0.0;array_K[2] = m_cam_intrinsic[4];
  array_K[3] = 0.0;array_K[4] = m_cam_intrinsic[3];array_K[5] = m_cam_intrinsic[5];
  array_K[6] = 0;array_K[7] = 0;array_K[8] = 1.0;
  boost::array<double,12> array_P;
  array_P[0] = m_cam_intrinsic[2];array_P[1] = 0.0;array_P[2] = m_cam_intrinsic[4]; array_P[3] = 0.0;
  array_P[4] = 0.0;array_P[5] = m_cam_intrinsic[3];array_P[6] = m_cam_intrinsic[5]; array_P[7] = 0.0;
  array_P[8] = 0;array_P[9] = 0;array_P[10] = 1.0; array_P[11] = 0.0;
  cam_info.K = array_K;
  cam_info.P = array_P;
  m_camera_model.fromCameraInfo(cam_info);

  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1e3;
  ROS_WARN("Camera info processed in %.3f [ms]", duration);
  m_cam_init = true;
}
// void mlcpp_class::load_mesh1()
// {
//   std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
//   //load ply
//   ROS_INFO("loading %s", m_meshfile.c_str());

//   pcl::PolygonMesh mesh;
//   if (pcl::io::loadPLYFile(m_meshfile, mesh) == -1){
//     PCL_ERROR("Couldn't read file ply\n");
//     return;
//   }

//   // 传到全局变量
//   m_mesh = std::move(mesh);
//   // 检查网格是否正确加载
//   if (m_mesh.polygons.empty() || m_mesh.cloud.data.empty()) {
//       PCL_ERROR("PLY file is empty or not a valid mesh!\n");
//       return;
//   }
//   auto t2 = std::chrono::high_resolution_clock::now();
//   double duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
//   ROS_INFO("PLY Mesh loaded in %.2f ms, faces: %zu", duration, m_mesh.polygons.size());
//   // ROS中显示网格
//   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
//   pcl::fromPCLPointCloud2(m_mesh.cloud, *cloud);
//   m_cloud_map = *cloud;  // 将点云数据存储到 m_cloud_map
//   sensor_msgs::PointCloud2 cloud_msg;
//   pcl::toROSMsg(*cloud, cloud_msg);  
//   cloud_msg.header.frame_id = "map";  
//   cloud_msg.header.stamp = ros::Time::now(); 
// }

// 可视化点云和mesh模型
void mlcpp_class::pointCloudandMeshViewer(const pcl::PolygonMesh& mesh)
{
	// 输出结果到可视化窗口
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D PointCloud Viewer"));

	// 显示重建点云
	int v1;
	viewer->createViewPort(0.0, 0.0, 1.0, 1.0, v1);  // 右侧窗口
	viewer->setBackgroundColor(0.0, 0.0, 0.0, v1);   // 黑色背景
	viewer->addText("mesh", 10, 10, "mesh_text", v1);
	viewer->addPolygonMesh(mesh, "mesh", v1);
	viewer->setRepresentationToWireframeForAllActors(); // 网格模型以线框图模式显示
	// 可视化循环
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}
}

void mlcpp_class::load_mesh()
{
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  // 将ply格式数据加载为PolygonMesh对象
  pcl::PolygonMesh poly_mesh;
  // 成功返回0，失败返回-1
  ROS_INFO("loading %s", m_meshfile.c_str());
  if(-1 == pcl::io::loadPLYFile(m_meshfile, poly_mesh)){
      std::cout<<"load ply file failed. please check it."<<std::endl;
      return ;
  }
  if(!poly_mesh.cloud.data.empty()){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(poly_mesh.cloud, *cloud);

    // 对网格进行降采样生成点云
    // pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::UniformSampling<pcl::PointXYZ> sampling;
    // sampling.setInputCloud(cloud);
    // sampling.setRadiusSearch(0.01);  // 设置采样半径
    // sampling.filter(*sampled_cloud);
    // 保存点云
    // pcl::io::savePLYFile("output_cloud.ply", *sampled_cloud); 

    // 直接使用原始点云进行可视化
    pcl::visualization::PCLVisualizer viewer("Mesh to Point Cloud");
    viewer.addPointCloud(cloud, "original_cloud");
    viewer.spin();

    m_cloud_map = *cloud;
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1e3;
    ROS_WARN("Map successfully obtained from PCD in %.3f [ms]", duration);
    m_pcd_load = true;
    // 直接mesh可视化
    // this->pointCloudandMeshViewer(poly_mesh);
  } 
}


// void mlcpp_class::load_pcd(){
//   std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
//   m_cloud_map.clear();
//   m_cloud_center.clear();
//   m_cloud_none_viewed.clear();
//   m_cloud_initial_view_point.clear();
//   m_optimized_view_point.clear();

//   ROS_INFO("loading %s", m_infile.c_str());
//   if (pcl::io::loadPCDFile<pcl::PointXYZ> (m_infile.c_str (), m_cloud_map) == -1) //* load the file
//   {
//     PCL_ERROR ("Couldn't read pcd file \n");
//     return;
//   }

//   std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
//   double duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1e3;

//   ROS_WARN("Map successfully obtained from PCD in %.3f [ms]", duration);

//   m_pcd_load = true;
//   //try
//   buildKDTree();
// }

void mlcpp_class::preprocess_pcd(){
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  ////// Ground Eleminate
  ROS_INFO("Ground filtering Start!");
  //#pragma omp parallel for
  m_pcd_center_point.x = 0; m_pcd_center_point.y = 0; m_pcd_center_point.z = 0;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_map_nogr(new pcl::PointCloud<pcl::PointXYZ>);
  for(size_t i = 0; i < m_cloud_map.points.size() ; ++i){
    pcl::PointXYZ point;
    point.x = m_cloud_map.points[i].x;
    point.y = m_cloud_map.points[i].y;
    point.z = m_cloud_map.points[i].z;
    if(point.z > 0.3){
      Eigen::Vector4d vec;
      vec<<point.x, point.y, point.z, 1;
      cloud_map_nogr->points.push_back(point);
      m_pcd_center_point.x += point.x;
      m_pcd_center_point.y += point.y;
      m_pcd_center_point.z += point.z;
    }
  }
  m_pcd_center_point.x = m_pcd_center_point.x / cloud_map_nogr->points.size();
  m_pcd_center_point.y = m_pcd_center_point.y / cloud_map_nogr->points.size();
  m_pcd_center_point.z = m_pcd_center_point.z / cloud_map_nogr->points.size();
  m_cloud_center.push_back(m_pcd_center_point);

  m_cloud_map.clear();
  m_cloud_map = *cloud_map_nogr;
  m_cloud_map.width = m_cloud_map.points.size();
  m_cloud_map.height = 1;
  ROS_INFO("Ground filtering Finished!");
  ////// Normal estimation
  ROS_INFO("Normal Estimation Start!");
  m_cloud_normals.clear();
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  *cloud_in = m_cloud_map;
  m_normal_estimator.setInputCloud(cloud_in);
  m_normal_estimator.setSearchMethod (tree);
  //m_normal_estimator.setKSearch (20);
  m_normal_estimator.setRadiusSearch (4); //TODO, parameterlize
  m_normal_estimator.compute (*cloud_normals);
  for (size_t i=0; i<cloud_normals->points.size(); ++i)
  {
    flip_normal(cloud_in->points[i], m_pcd_center_point, cloud_normals->points[i].normal[0], cloud_normals->points[i].normal[1], cloud_normals->points[i].normal[2]);
    pcl::PointNormal temp_ptnorm;
    temp_ptnorm.x = cloud_in->points[i].x;
    temp_ptnorm.y = cloud_in->points[i].y;
    temp_ptnorm.z = cloud_in->points[i].z;
    temp_ptnorm.normal[0] = cloud_normals->points[i].normal[0];
    temp_ptnorm.normal[1] = cloud_normals->points[i].normal[1];
    temp_ptnorm.normal[2] = cloud_normals->points[i].normal[2];
    m_cloud_normals.push_back(temp_ptnorm);
  }
  m_cloud_normals.width = m_cloud_normals.points.size();
  m_cloud_normals.height = 1;
  ROS_INFO("Normal Estimation Finish!");
  ROS_INFO("Cloud size : %lu",cloud_in->points.size());
  ROS_INFO("Cloud Normal size : %lu",cloud_normals->points.size());
  
  m_normal_pose_array = pclnormal_to_posearray(m_cloud_normals);
  m_normal_pose_array.header.frame_id = "map";

  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1e3;
  ROS_WARN("PCL preprocessed in %.3f [ms]", duration);

  m_pre_process=true;	
}

// 主要是针对点云做滤波和法线估计，
// 而 Mesh 本身已经包含了表面信息，我们只需要调整代码适应 Mesh：
void mlcpp_class::preprocess_ply()
{
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  
  // 计算mesh网格的法线
  // 先遍历提取mesh中的顶点，计算顶点的法线，在这一步判断是否是朝向外的法线，剔除或者反向，
  // 再对顶点法线进行加权平均归一化，得到面片的法线，这样确保面片法线朝向外部。 
  //TODO


  // 计算处理时间
  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1e3;
  ROS_WARN("Mesh preprocessing completed in %.3f [ms]", duration);
  m_pre_process=true;	
}

void mlcpp_class::calc_cb(const std_msgs::Empty::ConstPtr& msg){
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	if (m_cam_init && m_pcd_load && m_pre_process){
    m_cloud_initial_view_point.clear();
    m_cloud_none_viewed.clear();
    m_optimized_view_point.clear();
    m_all_layer_path.header.stamp = ros::Time::now();
    m_all_layer_path.header.frame_id = "map";
    m_all_layer_path.poses.clear();
    pcl::PointNormal prev_layer_endpoint;
    
    ////// calculate Coverage Path
    ///Make Initial viewpoint
    bool finish = false;
    int current_layer=1;
    pcl::PointNormal minpt;
    pcl::PointNormal maxpt;
    // 获取点云的最小、最大 3D 边界值
    pcl::getMinMax3D(m_cloud_normals, minpt, maxpt);
    float minpt_z = minpt.z;
    ///PCL Slice with Z axis value (INITIAL)
    pcl::PointCloud<pcl::PointNormal>::Ptr Sliced_ptnorm(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_pt_normals(new pcl::PointCloud<pcl::PointNormal>);
    // 拷贝原始点云数据
    *cloud_pt_normals = m_cloud_normals;
    pcl::PassThrough<pcl::PointNormal> pass;
    pass.setInputCloud (cloud_pt_normals);
    // 沿z轴过滤
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (minpt_z,minpt_z+m_slice_height);
    // 过滤后的点存入Sliced_ptnorm
    pass.filter (*Sliced_ptnorm);


    while(!finish)
    {
      // 遍历切片层级
      //!BUG 不能用基于高度切片的方式 对矮小的物体直接失效
      if (minpt_z+m_slice_height >= maxpt.z){ // Check if last layer
        finish = true;
      }
      ///PCL make viewpoints by points and normals
      pcl::PointCloud<pcl::PointNormal>::Ptr viewpoint_ptnorm(new pcl::PointCloud<pcl::PointNormal>);
      viewpoint_ptnorm->clear();
      for(int i=0;i<Sliced_ptnorm->points.size();i++)
      {
        pcl::PointNormal temp_ptnorm;
        // 视点由原点云沿法线方向偏移 m_view_pt_dist 个单位生成
        temp_ptnorm.x = Sliced_ptnorm->points[i].x + Sliced_ptnorm->points[i].normal[0] * m_view_pt_dist;
        temp_ptnorm.y = Sliced_ptnorm->points[i].y + Sliced_ptnorm->points[i].normal[1] * m_view_pt_dist;
        temp_ptnorm.z = Sliced_ptnorm->points[i].z + Sliced_ptnorm->points[i].normal[2] * m_view_pt_dist;
        if(temp_ptnorm.z <= 0 ) continue;
        // 反转法线,方向朝原点云的方向
        temp_ptnorm.normal[0] = -Sliced_ptnorm->points[i].normal[0];
        temp_ptnorm.normal[1] = -Sliced_ptnorm->points[i].normal[1];
        temp_ptnorm.normal[2] = -Sliced_ptnorm->points[i].normal[2];
        viewpoint_ptnorm->push_back(temp_ptnorm);
      }
      ///PCL downsample viewpoints with VoxelGrid
      // pcl体素滤波,减少点云密度,降低计算开销
      pcl::VoxelGrid<pcl::PointNormal> voxgrid;
      pcl::PointCloud<pcl::PointNormal>::Ptr Voxed_Sliced_Viewpt (new pcl::PointCloud<pcl::PointNormal>);
      Voxed_Sliced_Viewpt->clear();
      pcl::PointCloud<pcl::PointNormal>::Ptr Voxed_Sliced_Admitted_Viewpt (new pcl::PointCloud<pcl::PointNormal>);
      Voxed_Sliced_Admitted_Viewpt->clear();
      voxgrid.setInputCloud(viewpoint_ptnorm);
      // 点云变成体素 2*2*2*0.8
      voxgrid.setLeafSize(m_view_pt_each_dist,m_view_pt_each_dist,m_view_pt_each_dist*0.8);
      voxgrid.filter(*Voxed_Sliced_Viewpt);

      ROS_WARN("current layer %d, not viewed, voxelized initial viewpoints: %d", current_layer, Voxed_Sliced_Viewpt->points.size());
      for (int idx = 0; idx < Voxed_Sliced_Viewpt->points.size(); ++idx)
      {
        pcl::PointXYZ initial_view_pts;
        initial_view_pts.x = Voxed_Sliced_Viewpt->points[idx].x;
        initial_view_pts.y = Voxed_Sliced_Viewpt->points[idx].y;
        initial_view_pts.z = Voxed_Sliced_Viewpt->points[idx].z;
        // 处理好的点
        m_cloud_initial_view_point.push_back(initial_view_pts);
      }
      //TODO 去掉随机性,按顺序暴力遍历
      // 随机筛选视角点
      pcl::PointCloud<pcl::PointNormal>::Ptr Sliced_ptnorm_Unview (new pcl::PointCloud<pcl::PointNormal>);
      Sliced_ptnorm_Unview->clear();
      pcl::copyPointCloud(*Sliced_ptnorm,*Sliced_ptnorm_Unview);
      ///PCL downsample viewpoints by view calculation
      int admitted=0;
      //?对 Voxed_Sliced_Viewpt 点云的索引进行随机打乱
      int a[Voxed_Sliced_Viewpt->points.size()];
      for(int i=0;i<Voxed_Sliced_Viewpt->points.size();i++){
        a[i]=i;
      }
      //！TODO 去掉随机性会发生什么
      // 通过 random_shuffle 打乱索引，随机挑选点来优化分布
      // random_shuffle(&a[0], &a[Voxed_Sliced_Viewpt->points.size()]);
      for(int k=0;k<Voxed_Sliced_Viewpt->points.size();k++)
      {
        int i = a[k];
        vector<int> toerase;
        vector<int> view_comp_map;
        Eigen::VectorXd viewpt(5);
        // 这里 viewpt 是一个 5 维向量，它的含义：
        // 索引	含义
        // 0	x 坐标
        // 1	y 坐标
        // 2	z 坐标
        // 3	俯仰角（Pitch）
        // 4	横滚角（Roll）
        //? 为什么不加偏航角?
        viewpt << Voxed_Sliced_Viewpt->points[i].x,Voxed_Sliced_Viewpt->points[i].y,Voxed_Sliced_Viewpt->points[i].z,
                asin(-Voxed_Sliced_Viewpt->points[i].normal[2])/M_PI*180.0,
                asin(Voxed_Sliced_Viewpt->points[i].normal[1]/cos(-Voxed_Sliced_Viewpt->points[i].normal[2]))/M_PI*180.0;
        // openmp多线程并行加速
        #pragma omp parallel for
        // 检查哪些点可以被当前视角点覆盖，并记录需要删除的索引（toerase）
        for(int j=0;j<Sliced_ptnorm_Unview->points.size();j++)
        {
          pcl::PointXYZ point_toview(Sliced_ptnorm_Unview->points[j].x,Sliced_ptnorm_Unview->points[j].y,
                                     Sliced_ptnorm_Unview->points[j].z);
          pcl::Normal point_normal(Sliced_ptnorm_Unview->points[j].normal[0],
                                   Sliced_ptnorm_Unview->points[j].normal[1],
                                   Sliced_ptnorm_Unview->points[j].normal[2]);
          if(check_cam_in(viewpt,point_toview,point_normal))
          {
            #pragma omp critical
            toerase.push_back(j);
          }
        }
        #pragma omp parallel for
        // 遍历 Sliced_ptnorm 原始点云，记录哪些点属于当前视角的可视区域（view_comp_map）
        for (size_t j=0;j<Sliced_ptnorm->points.size();j++)
        {
          pcl::PointXYZ point_toview(Sliced_ptnorm->points[j].x,
                                     Sliced_ptnorm->points[j].y,
                                     Sliced_ptnorm->points[j].z);
          pcl::Normal point_normal(Sliced_ptnorm->points[j].normal[0],
                                   Sliced_ptnorm->points[j].normal[1],
                                   Sliced_ptnorm->points[j].normal[2]);
          if(check_cam_in(viewpt,point_toview,point_normal))
          {
            #pragma omp critical
            view_comp_map.push_back(j);
          }
        }
        // 如果 view_comp_map 的大小乘以 m_view_overlap 仍然小于 toerase 的大小，
        // 则认为该可视点有效，将 toerase 中的点从 Sliced_ptnorm_Unview 删除，
        // 并将该可视点添加到 Voxed_Sliced_Admitted_Viewpt
        if(view_comp_map.size()*m_view_overlap < toerase.size()) // view_comp_map * 0.1 < toerase
        {
          sort(toerase.begin(),toerase.end());
          for(int j=toerase.size()-1;j>-1;j--)
          {
            Sliced_ptnorm_Unview->points.erase(Sliced_ptnorm_Unview->points.begin()+toerase[j]);
          }
          if (m_debug_mode){
            ROS_INFO("%d Point Left", Sliced_ptnorm_Unview->points.size());
            ROS_INFO("Viewpoint %d / %d Admitted ", i, Voxed_Sliced_Viewpt->points.size());
          }
          admitted++;
          Voxed_Sliced_Admitted_Viewpt->push_back(Voxed_Sliced_Viewpt->points[i]);
        }
        //else cout<<"Viewpoint "<<i<<"/"<<Voxed_Sliced_Viewpt->points.size()<<" Not Admitted"<<endl;
      }
      if (m_debug_mode){
        ROS_INFO("Admitted Viewpoint: %d", admitted);
      }
      ROS_WARN("current layer %d, still none-viewed points: %d among %d in slice", current_layer, Sliced_ptnorm_Unview->points.size(), Sliced_ptnorm->points.size());
      for (int idx = 0; idx < Sliced_ptnorm_Unview->points.size(); ++idx)
      {
        pcl::PointXYZ none_view_pcl;
        none_view_pcl.x = Sliced_ptnorm_Unview->points[idx].x;
        none_view_pcl.y = Sliced_ptnorm_Unview->points[idx].y;
        none_view_pcl.z = Sliced_ptnorm_Unview->points[idx].z;
        m_cloud_none_viewed.push_back(none_view_pcl);
      }

      
      if (Voxed_Sliced_Admitted_Viewpt->points.size() > 0){
        /// Solve TSP among downsampled viewpoints
        OrdreringTSP(Voxed_Sliced_Admitted_Viewpt);
        double best_distance = TwoOptTSP(Voxed_Sliced_Admitted_Viewpt);

        if (current_layer>1){        
          Eigen::Vector3d current_first(Voxed_Sliced_Admitted_Viewpt->points[0].x, Voxed_Sliced_Admitted_Viewpt->points[0].y, Voxed_Sliced_Admitted_Viewpt->points[0].z);
          Eigen::Vector3d current_end(Voxed_Sliced_Admitted_Viewpt->points[Voxed_Sliced_Admitted_Viewpt->points.size()-1].x, Voxed_Sliced_Admitted_Viewpt->points[Voxed_Sliced_Admitted_Viewpt->points.size()-1].y, Voxed_Sliced_Admitted_Viewpt->points[Voxed_Sliced_Admitted_Viewpt->points.size()-1].z) ;
          Eigen::Vector3d prev_layer_endpoint_points(prev_layer_endpoint.x, prev_layer_endpoint.y, prev_layer_endpoint.z);
          Eigen::Vector3d prev_layer_endpoint_normal(prev_layer_endpoint.normal[0], prev_layer_endpoint.normal[1], prev_layer_endpoint.normal[2]);

          //// forward
          if ( fabs( (current_first-prev_layer_endpoint_points).normalized().dot(prev_layer_endpoint_normal.normalized()) ) < 
            fabs( (current_end-prev_layer_endpoint_points).normalized().dot(prev_layer_endpoint_normal.normalized()) ) )
          {
            for(int idx=0; idx<Voxed_Sliced_Admitted_Viewpt->points.size(); ++idx)
            {
              pcl::PointXYZ optimized_viewpt;
              optimized_viewpt.x = Voxed_Sliced_Admitted_Viewpt->points[idx].x;
              optimized_viewpt.y = Voxed_Sliced_Admitted_Viewpt->points[idx].y;
              optimized_viewpt.z = Voxed_Sliced_Admitted_Viewpt->points[idx].z;
              m_optimized_view_point.push_back(optimized_viewpt);
              m_all_layer_path.poses.push_back(single_pclnormal_to_posestamped(Voxed_Sliced_Admitted_Viewpt->points[idx]));
            }
            prev_layer_endpoint = Voxed_Sliced_Admitted_Viewpt->points[Voxed_Sliced_Admitted_Viewpt->points.size()-1];
          }
          //// reverse
          else
          {
            for(int idx=Voxed_Sliced_Admitted_Viewpt->points.size()-1; idx>=0; --idx)
            {
              pcl::PointXYZ optimized_viewpt;
              optimized_viewpt.x = Voxed_Sliced_Admitted_Viewpt->points[idx].x;
              optimized_viewpt.y = Voxed_Sliced_Admitted_Viewpt->points[idx].y;
              optimized_viewpt.z = Voxed_Sliced_Admitted_Viewpt->points[idx].z;
              m_optimized_view_point.push_back(optimized_viewpt);
              m_all_layer_path.poses.push_back(single_pclnormal_to_posestamped(Voxed_Sliced_Admitted_Viewpt->points[idx]));
            }
            prev_layer_endpoint = Voxed_Sliced_Admitted_Viewpt->points[0];
          }
        }
        else{
          for(int idx=0; idx<Voxed_Sliced_Admitted_Viewpt->points.size(); ++idx)
          {
            pcl::PointXYZ optimized_viewpt;
            optimized_viewpt.x = Voxed_Sliced_Admitted_Viewpt->points[idx].x;
            optimized_viewpt.y = Voxed_Sliced_Admitted_Viewpt->points[idx].y;
            optimized_viewpt.z = Voxed_Sliced_Admitted_Viewpt->points[idx].z;
            m_optimized_view_point.push_back(optimized_viewpt);
            m_all_layer_path.poses.push_back(single_pclnormal_to_posestamped(Voxed_Sliced_Admitted_Viewpt->points[idx]));
          }
          prev_layer_endpoint = Voxed_Sliced_Admitted_Viewpt->points[Voxed_Sliced_Admitted_Viewpt->points.size()-1];
        }
        ROS_WARN("current layer %d, TSP %d points, leng: %.2f m", current_layer, Voxed_Sliced_Admitted_Viewpt->points.size(), best_distance);
      }
      else {
        ROS_WARN("current layer %d, no admitted points, skipping", current_layer);
      }

      ///PCL Slice with Z axis value (untill maxpt.z)
      minpt_z += m_slice_height;
      pass.setFilterFieldName ("z");
      pass.setFilterLimits (minpt_z,minpt_z+m_slice_height);
      pass.filter (*Sliced_ptnorm);
      current_layer++;
    } //while end
	} //if end
	else{
		ROS_WARN("One of cam info / PCD file loading / PCD pre-process has not been done yet");
	}

  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1e3;
  ROS_WARN("MLCPP calculation: %.3f [ms], path len: %.3f", duration, path_length(m_all_layer_path));
}

void mlcpp_class::visualizer_timer_func(const ros::TimerEvent& event){
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	if (m_cam_init && m_pcd_load && m_pre_process){
		m_cloud_map_pub.publish(cloud2msg(m_cloud_map));
    m_cloud_center_pub.publish(cloud2msg(m_cloud_center));
    m_cloud_normal_pub.publish(m_normal_pose_array);
    m_initial_view_point_pub.publish(cloud2msg(m_cloud_initial_view_point));
    m_cloud_none_viewed_pub.publish(cloud2msg(m_cloud_none_viewed));
    m_optimized_view_point_pub.publish(cloud2msg(m_optimized_view_point));
    m_all_layer_path_pub.publish(m_all_layer_path);
    // m_pubCloud.publish(m_mesh_to_pointcloud);

	}

  if(m_debug_mode){
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1e3;
    ROS_WARN("visualizing in %.2f [ms]", duration);
  }
}






///////// methods
//TODO: flip_normal not from center but from real view point, where recorded PCL for none-convex targets
void mlcpp_class::flip_normal(pcl::PointXYZ base, pcl::PointXYZ center, 
                              float & nx, float & ny, float & nz)
{
  float xdif = base.x - center.x;
  float ydif = base.y - center.y;
  if(xdif * nx + ydif * ny <0)
  {
    nx = -nx;
    ny = -ny;
    nz = -nz;
  }
}

void mlcpp_class::buildKDTree() 
{
    if (m_cloud_map.empty()) {
        ROS_WARN("Mesh 为空，无法构建 KD-Tree");
        return;
    }

    kdtree.setInputCloud(m_cloud_map.makeShared());
    ROS_INFO("KD-Tree 构建完成，点数: %lu", m_cloud_map.size());
}

// 可视点碰撞检测
// 检查视点是否与点云的某个点相交
bool mlcpp_class::check_viewpoint_collision(Eigen::VectorXd view_point_xyzpy, pcl::PointXYZ point, pcl::Normal normal)
{
  //TODO: check collision with other viewpoints
  //TODO: check collision with obstacles
  pcl::PointXYZ view_point(view_point_xyzpy(0), view_point_xyzpy(1), view_point_xyzpy(2));
  // **使用 KD-Tree 限制搜索范围**
  std::vector<int> indices;
  std::vector<float> distances;
  //TODO parameterlize
  float search_radius = 5.0;  // 设定搜索半径，单位：米，可调
  kdtree.radiusSearch(view_point, search_radius, indices, distances, 1000);
  // 遍历 KD-Tree 找到的最近邻点
  for (size_t i = 0; i < indices.size(); i += 3) {
      if (i + 2 >= indices.size()) break;
      const auto& v0 = m_cloud_map.points[indices[i]];
      const auto& v1 = m_cloud_map.points[indices[i+1]];
      const auto& v2 = m_cloud_map.points[indices[i+2]];
      
      // 检查三角形是否退化
      Eigen::Vector3d e1(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
      Eigen::Vector3d e2(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
      if (e1.cross(e2).norm() < 1e-6) continue; // 跳过退化的三角形
      
      if (isIntersecting(view_point, point, v0, v1, v2)) {
          return true;  // true表示存在碰撞
      }
  }
  // 反转最终返回值逻辑
  return !check_cam_in(view_point_xyzpy, point, normal); // 原直接返回check_cam_in结果
}

// 这是一个用于验证视点有效性的核心函数，主要包含三个约束条件的检查：投影约束、距离约束和角度约束。
// 投影约束：检查点是否在相机的视野范围内。
// 距离约束：检查点与视点的距离是否在给定的最大距离范围内。
// 角度约束：检查点的法线与视点的夹角是否在给定的最大角度范围内。
// 函数返回一个布尔值，表示点是否满足所有约束条件。
//TODO: try to add constrain about two viewpoints cant across every mesh voxel
bool mlcpp_class::check_cam_in(Eigen::VectorXd view_point_xyzpy,pcl::PointXYZ point,pcl::Normal normal)
{
  Eigen::Vector3d pt_bef_rot(point.x-view_point_xyzpy(0),point.y-view_point_xyzpy(1),point.z-view_point_xyzpy(2));
  Eigen::Vector3d pt_aft_rot = RPYtoR(0,-view_point_xyzpy(3),-view_point_xyzpy(4))*pt_bef_rot;
  Eigen::Vector4d pt_cvv(pt_aft_rot(0),pt_aft_rot(1),pt_aft_rot(2),1);
  Eigen::Matrix4d view_pt;
  view_pt.setIdentity();
  view_pt.block<3,3>(0,0) = RPYtoR(-90,0,-90);
  Eigen::Vector4d new_pt = view_pt.inverse() * pt_cvv;
  cv::Point3d pt_cv(new_pt(0), new_pt(1), new_pt(2));
  cv::Point2d uv;
  uv = m_camera_model.project3dToPixel(pt_cv);
  uv.x = floor(abs(uv.x)) * ((uv.x > 0) - (uv.x < 0));
  uv.y = floor(abs(uv.y)) * ((uv.y > 0) - (uv.y < 0));
  // 约束1：投影约束（检查是否在成像范围内）
  if(uv.x<0 || uv.x>m_cam_intrinsic[0] || uv.y<0 || uv.y>m_cam_intrinsic[1]) return false;
  // 约束2：距离约束（检查最大有效观测距离）
  float dist = sqrt(pow((view_point_xyzpy(0)-point.x),2)+pow((view_point_xyzpy(1)-point.y),2)+
                    pow((view_point_xyzpy(2)-point.z),2));
  if(dist>m_max_dist) return false;
  // 约束3：角度约束（检查法线方向与观测方向的夹角）
  Eigen::Vector3d normal_pt(normal.normal_x,normal.normal_y,normal.normal_z);
  Eigen::Vector3d Normal_view_pt((view_point_xyzpy(0)-point.x)/dist,(view_point_xyzpy(1)-point.y)/dist,
                                    (view_point_xyzpy(2)-point.z)/dist);
  double inner_product = Normal_view_pt.dot(normal_pt);
  double angle = acos(inner_product)/M_PI*180.0;
  if(abs(angle)>m_max_angle) return false;
  return true;
}

Eigen::Matrix3d mlcpp_class::RPYtoR(double roll,double pitch,double yaw)
{
  Eigen::Matrix3d Rmatrix;
  Eigen::Matrix3d Rmatrix_y;
  Eigen::Matrix3d Rmatrix_p;
  Eigen::Matrix3d Rmatrix_r;
  yaw = yaw*M_PI/180.0;
  roll = roll*M_PI/180.0;
  pitch = pitch*M_PI/180.0;
  Rmatrix_y << cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1;
  Rmatrix_p << cos(pitch), 0, sin(pitch), 0, 1, 0, -sin(pitch), 0, cos(pitch);
  Rmatrix_r << 1, 0, 0, 0, cos(roll), -sin(roll), 0, sin(roll), cos(roll);
  Rmatrix = Rmatrix_y * Rmatrix_p * Rmatrix_r;
  return Rmatrix;
}

//TWO OPT ALGORITHM
void mlcpp_class::TwoOptSwap(pcl::PointCloud<pcl::PointNormal>::Ptr pclarray,int start,int finish)
{
  int size = pclarray->points.size();
  pcl::PointCloud<pcl::PointNormal> temp_Array;
  for(int i=0;i<=start-1;i++) temp_Array.push_back(pclarray->points[i]);
  for(int i=finish;i>=start;i--) temp_Array.push_back(pclarray->points[i]);
  for(int i=finish+1;i<=size-1;i++) temp_Array.push_back(pclarray->points[i]);
  pcl::copyPointCloud(temp_Array,*pclarray);
}

double mlcpp_class::PclArrayCost(pcl::PointCloud<pcl::PointNormal>::Ptr pclarray, double &distance)
{
  double cost = 0;
  double distance_out = 0;
  for(int i=1;i<pclarray->points.size();i++)
  {
    Eigen::Vector3d current(pclarray->points[i-1].x, pclarray->points[i-1].y, pclarray->points[i-1].z);
    Eigen::Vector3d current_normal(pclarray->points[i-1].normal[0], pclarray->points[i-1].normal[1], pclarray->points[i-1].normal[2]);
    Eigen::Vector3d next(pclarray->points[i].x, pclarray->points[i].y, pclarray->points[i].z);
    Eigen::Vector3d direction_vector = next - current;
    double dist = direction_vector.norm();
    // 总距离
    distance_out += dist;
    // 点的法向量变化越大，惩罚越大，避免路径突然偏离
    cost = cost + dist + dist*fabs(direction_vector.normalized().dot(current_normal.normalized())); // penalty on penetrating through pcd target
  }
  distance = distance_out;
  return cost;
}

double mlcpp_class::TwoOptTSP(pcl::PointCloud<pcl::PointNormal>::Ptr pclarray)
{
  pcl::PointCloud<pcl::PointNormal> temp_Array;
  pcl::copyPointCloud(*pclarray,temp_Array);
  int size = pclarray->points.size();
  int improve = 0;
  double best_distance = 0.0;
  double best_cost = PclArrayCost(pclarray, best_distance);
  if (m_debug_mode){
    ROS_INFO("Initial distance: %.2f", best_cost);
  }
  while (improve<m_TSP_trial) // 100次迭代
  {
    for ( int i = 1; i <size - 2; i++ )
    {
      for ( int k = i + 1; k < size-2; k++)
      {
        TwoOptSwap( pclarray, i,k );
        double new_distance = 0.0;
        double new_cost = PclArrayCost(pclarray, new_distance);
        if ( new_cost < best_cost )
        {
          improve = 0;
          pcl::copyPointCloud(*pclarray,temp_Array);
          best_distance = new_distance;
          best_cost = new_cost;
        }
      }
    }
    improve ++;
  }
  pcl::copyPointCloud(temp_Array,*pclarray);
  if (m_debug_mode){
    ROS_INFO("Final distance: %.2f, cost: %.2f", best_distance, best_cost);
    ROS_INFO("TwoOptTSP Finished");
  }
  return best_distance;
}

void mlcpp_class::OrdreringTSP(pcl::PointCloud<pcl::PointNormal>::Ptr pclarray)
{
  pcl::PointCloud<pcl::PointNormal> temp_Array;
  pcl::PointNormal minpt;
  pcl::PointNormal maxpt;
  // 1. 获取点云的Z轴极小极大值
  pcl::getMinMax3D(*pclarray,minpt,maxpt);
  // cout<<minpt.z<<"<-min,max->"<<maxpt.z<<endl;
  for(int i=0;i<pclarray->points.size();i++)
  {
    if(pclarray->points[i].z == minpt.z)
    {
      temp_Array.push_back(pclarray->points[i]);
      // cout<<pclarray->points[i].z<<endl;
      for(int j=0;j<pclarray->points.size();j++)
      {
        if(pclarray->points[j].z == maxpt.z)
        {
          // cout<<pclarray->points[j].z<<endl;
          for(int k=0;k<pclarray->points.size();k++)
          {
            if(k!=i && k!=j) temp_Array.push_back(pclarray->points[k]);
          }
          // 6. 将Z轴最大值点作为路径终点
          temp_Array.push_back(pclarray->points[j]);
          break;
        }
      }
      break;
    }
  }
  // 7. 用新排序替换原始点云
  pcl::copyPointCloud(temp_Array,*pclarray);
}



#endif
