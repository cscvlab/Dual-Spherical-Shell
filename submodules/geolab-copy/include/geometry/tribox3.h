/**
 * @file tribox3.h
 * @brief Make some change that fit my work
 * @date 2022-05-09
 * @copyright Copyright (c) 2022
 */

/* This file is part of PyMesh. 
/* Copyright (c) 2015 by Qingnan Zhou */
/********************************************************/
/* AABB-triangle overlap test code                      */
/* by Tomas Akenine-M�ller                              */
/* Function: int triBoxOverlap(Float boxcenter[3],      */
/*          Float boxhalfsize[3],Float triverts[3][3]); */
/* History:                                             */
/*   2001-03-05: released the code in its first version */
/*   2001-06-18: changed the order of the tests, faster */
/*                                                      */
/* Acknowledgement: Many thanks to Pierre Terdiman for  */
/* suggestions and discussions on how to optimize code. */
/* Thanks to David Hunt for finding a ">="-bug!         */
/********************************************************/
// Code taken from
// http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox3.txt
// on Aug 20, 2014
#pragma once
#ifndef TRIBOX3_H
#define TRIBOX3_H
#include<Eigen/Eigen>
#include <math.h>
#include <stdio.h>

#define X 0
#define Y 1
#define Z 2

#define CROSS(dest,v1,v2) \
          dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
          dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
          dest[2]=v1[0]*v2[1]-v1[1]*v2[0]; 

#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])

#define SUB(dest,v1,v2) \
          dest[0]=v1[0]-v2[0]; \
          dest[1]=v1[1]-v2[1]; \
          dest[2]=v1[2]-v2[2]; 

#define FINDMINMAX(x0,x1,x2,min,max) \
  min = max = x0;   \
  if(x1<min) min=x1;\
  if(x1>max) max=x1;\
  if(x2<min) min=x2;\
  if(x2>max) max=x2;

inline int planeBoxOverlap(const Eigen::Vector3f normal, const Eigen::Vector3f vert, const float voxel_half_size)	// -NJMP-
{
  int q;
  Eigen::Vector3f vmin, vmax;
  float v;
//   Float vmin[3],vmax[3],v;
  for(q=0;q<=2;q++)
  {
    v=vert[q];					// -NJMP-
    if(normal[q]>0.0f)
    {
      vmin[q]=-voxel_half_size - v;	// -NJMP-
      vmax[q]= voxel_half_size - v;	// -NJMP-
    }
    else
    {
      vmin[q]= voxel_half_size - v;	// -NJMP-
      vmax[q]=-voxel_half_size - v;	// -NJMP-
    }
  }
  if(DOT(normal,vmin)>0.0f) return 0;	// -NJMP-
  if(DOT(normal,vmax)>=0.0f) return 1;	// -NJMP-
  
  return 0;
}


/*======================== X-tests ========================*/
#define AXISTEST_X01(a, b, fa, fb)			   \
	p0 = a*A.y() - b*A.z();			       	   \
	p2 = a*C.y() - b*C.z();			       	   \
        if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} \
	rad = fa * voxel_half_size + fb * voxel_half_size;   \
	if(min>rad || max<-rad) return 0;

#define AXISTEST_X2(a, b, fa, fb)			   \
	p0 = a*A.y() - b*A.z();			           \
	p1 = a*B.y() - b*B.z();			       	   \
        if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
	rad = fa * voxel_half_size + fb * voxel_half_size;   \
	if(min>rad || max<-rad) return 0;

/*======================== Y-tests ========================*/
#define AXISTEST_Y02(a, b, fa, fb)			   \
	p0 = -a*A.x() + b*A.z();		      	   \
	p2 = -a*C.x() + b*C.z();	       	       	   \
        if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} \
	rad = fa * voxel_half_size + fb * voxel_half_size;   \
	if(min>rad || max<-rad) return 0;

#define AXISTEST_Y1(a, b, fa, fb)			   \
	p0 = -a*A.x() + b*A.z();		      	   \
	p1 = -a*B.x() + b*B.z();	     	       	   \
        if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
	rad = fa * voxel_half_size + fb * voxel_half_size;   \
	if(min>rad || max<-rad) return 0;

/*======================== Z-tests ========================*/

#define AXISTEST_Z12(a, b, fa, fb)			   \
	p1 = a*B.x() - b*B.y();			           \
	p2 = a*C.x() - b*C.y();			       	   \
        if(p2<p1) {min=p2; max=p1;} else {min=p1; max=p2;} \
	rad = fa * voxel_half_size + fb * voxel_half_size;   \
	if(min>rad || max<-rad) return 0;

#define AXISTEST_Z0(a, b, fa, fb)			   \
	p0 = a*A.x() - b*A.y();				   \
	p1 = a*B.x() - b*B.y();			           \
        if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
	rad = fa * voxel_half_size + fb * voxel_half_size;   \
	if(min>rad || max<-rad) return 0;

inline bool triangle_test(const Eigen::Vector3f p, const Eigen::Matrix3f tri, float voxel_half_size){
    Eigen::Vector3f A, B, C, AB, BC, CA;
    float min, max, p0, p1, p2, rad, fex, fey, fez;
    Eigen::Vector3f normal;

    A = tri.row(0).transpose() - p;
    B = tri.row(1).transpose() - p;
    C = tri.row(2).transpose() - p;

    AB = B - A;
    BC = C - B;
    CA = A - C;

    /* Bullet 3:  */
    /*  test the 9 tests first (this was faster) */
    fex = std::abs(AB.x());
    fey = std::abs(AB.y());
    fez = std::abs(AB.z());

    AXISTEST_X01(AB.z(), AB.y(), fez, fey);
    AXISTEST_Y02(AB.z(), AB.x(), fez, fex);
    AXISTEST_Z12(AB.y(), AB.x(), fey, fex);

    fex = std::abs(BC.x());
    fey = std::abs(BC.y());
    fez = std::abs(BC.z());
    AXISTEST_X01(BC.z(), BC.y(), fez, fey);
    AXISTEST_Y02(BC.z(), BC.x(), fez, fex);
    AXISTEST_Z0(BC.y(), BC.x(), fey, fex);

    fex = std::abs(CA.x());
    fey = std::abs(CA.y());
    fez = std::abs(CA.z());
    AXISTEST_X2(CA.z(), CA.y(), fez, fey);
    AXISTEST_Y1(CA.z(), CA.x(), fez, fex);
    AXISTEST_Z12(CA.y(), CA.x(), fey, fex);

    /* Bullet 1: */
    /*  first test overlap in the {x,y,z}-directions */
    /*  find min, max of the triangle each direction, and test for overlap in */
    /*  that direction -- this is equivalent to testing a minimal AABB around */
    /*  the triangle against the AABB */

    /* test in X-direction */
    FINDMINMAX(A.x(),B.x(),C.x(),min,max);
    if(min>voxel_half_size || max<-voxel_half_size) return 0;

    /* test in Y-direction */
    FINDMINMAX(A.y(),B.y(),C.y(),min,max);
    if(min>voxel_half_size || max<-voxel_half_size) return 0;

    /* test in Z-direction */
    FINDMINMAX(A.z(),B.z(),C.z(),min,max);
    if(min>voxel_half_size || max<-voxel_half_size) return 0;

    /* Bullet 2: */
    /*  test if the box intersects the plane of the triangle */
    /*  compute plane equation of triangle: normal*x+d=0 */
    normal = AB.cross(BC);
    // CROSS(normal,e0,e1);
    // -NJMP- (line removed here)
    if(!planeBoxOverlap(normal,A,voxel_half_size)) return 0;	// -NJMP-

    return 1;   /* box and triangle overlaps */
}

inline bool triangle_test(const Eigen::Vector3f p, Eigen::Vector3f A, Eigen::Vector3f B, Eigen::Vector3f C, float voxel_half_size){
    Eigen::Vector3f AB, BC, CA;
    float min, max, p0, p1, p2, rad, fex, fey, fez;
    Eigen::Vector3f normal;

    A -= p;
    B -= p;
    C -= p;

    AB = B - A;
    BC = C - B;
    CA = A - C;

    /* Bullet 3:  */
    /*  test the 9 tests first (this was faster) */
    fex = std::abs(AB.x());
    fey = std::abs(AB.y());
    fez = std::abs(AB.z());

    AXISTEST_X01(AB.z(), AB.y(), fez, fey);
    AXISTEST_Y02(AB.z(), AB.x(), fez, fex);
    AXISTEST_Z12(AB.y(), AB.x(), fey, fex);

    fex = std::abs(BC.x());
    fey = std::abs(BC.y());
    fez = std::abs(BC.z());
    AXISTEST_X01(BC.z(), BC.y(), fez, fey);
    AXISTEST_Y02(BC.z(), BC.x(), fez, fex);
    AXISTEST_Z0(BC.y(), BC.x(), fey, fex);

    fex = std::abs(CA.x());
    fey = std::abs(CA.y());
    fez = std::abs(CA.z());
    AXISTEST_X2(CA.z(), CA.y(), fez, fey);
    AXISTEST_Y1(CA.z(), CA.x(), fez, fex);
    AXISTEST_Z12(CA.y(), CA.x(), fey, fex);

    /* Bullet 1: */
    /*  first test overlap in the {x,y,z}-directions */
    /*  find min, max of the triangle each direction, and test for overlap in */
    /*  that direction -- this is equivalent to testing a minimal AABB around */
    /*  the triangle against the AABB */

    /* test in X-direction */
    FINDMINMAX(A.x(),B.x(),C.x(),min,max);
    if(min>voxel_half_size || max<-voxel_half_size) return 0;

    /* test in Y-direction */
    FINDMINMAX(A.y(),B.y(),C.y(),min,max);
    if(min>voxel_half_size || max<-voxel_half_size) return 0;

    /* test in Z-direction */
    FINDMINMAX(A.z(),B.z(),C.z(),min,max);
    if(min>voxel_half_size || max<-voxel_half_size) return 0;

    /* Bullet 2: */
    /*  test if the box intersects the plane of the triangle */
    /*  compute plane equation of triangle: normal*x+d=0 */
    normal = AB.cross(BC);
    // CROSS(normal,e0,e1);
    // -NJMP- (line removed here)
    if(!planeBoxOverlap(normal,A,voxel_half_size)) return 0;	// -NJMP-

    return 1;   /* box and triangle overlaps */
}

#undef X
#undef Y
#undef Z
#undef CROSS
#undef DOT
#undef SUB
#undef FINDMINMAX
#undef AXISTEST_X01
#undef AXISTEST_X2
#undef AXISTEST_Y02
#undef AXISTEST_Y1
#undef AXISTEST_Z12
#undef AXISTEST_Z0

#endif
