#pragma once

#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <opencv2\opencv.hpp>
#include <Eigen\Eigen>
#include <OpenMesh\Core\IO\MeshIO.hh>
#include <OpenMesh\Core\Mesh\TriMesh_ArrayKernelT.hh>

using namespace std;
using namespace cv;
using namespace Eigen;

typedef OpenMesh::TriMesh_ArrayKernelT<> TriMesh;

void readTexture(char *path,int num,int frameinterval,vector<Mat>&textureimg)
{
	cout<<"reading texture..."<<endl;
	for(int i=0;i<num;i=i+frameinterval)
	{
		char buffer[100];
		sprintf(buffer,"%s/image/frame%03d.png",path,i);
		Mat curimg = imread(string(buffer));
		flip(curimg,curimg,1);
		textureimg.push_back(curimg);
	}
}

void readCamera(char *path,int num,int frameinterval,vector<Matrix4f,aligned_allocator<Matrix4f>>&external)
{
	cout<<"reading camera..."<<endl;
	for(int i=0;i<num;i=i+frameinterval)
	{
		char buffer[100];
		sprintf(buffer,"%s/Geometry/frame%03d.txt",path,i);
		Matrix4f temp;
		ifstream fin(buffer);
		for(int y=0;y<4;y++)
		{
			for(int x=0;x<4;x++)
				fin>>temp(y,x);
		}
		temp(3,3) = 1.0;
		external.push_back(temp.transpose());
	}
}

void readDepth(char *path,int num,int frameinterval,vector<vector<float>>&depthdata)
{
	cout<<"reading depth..."<<endl;
	for(int i=0;i<num;i=i+frameinterval)
	{
		char buffer[100];
		sprintf(buffer,"%s/depth/depth_denoise%03d.txt",path,i);
		ifstream fin(buffer);
		vector<float>curdepth;
		float temp;
		for(int i=0;i<IMGWIDTH*IMGHEIGHT;i++)
		{
			fin>>temp;
			curdepth.push_back(temp);
		}
		fin.close();
		depthdata.push_back(curdepth);
	}
}

bool isValid(Vector2i imgPoint,int width,int height)
{
	if(imgPoint[0]>=0&&imgPoint[0]<width&&imgPoint[1]>=0&&imgPoint[1]<height)
		return true;
	else
		return false;
}

void ProjectFromWorldToImage(Vector3f src,Vector2i &dst,Matrix4f intrinsic,Matrix4f external)
{
	Vector4f world_homo(src[0],src[1],src[2],1.0);
	Vector4f img_homo = intrinsic*external*world_homo;
	dst[0] = static_cast<int>(img_homo[0]/img_homo[2]);
	dst[1] = static_cast<int>(img_homo[1]/img_homo[2]);
}

void ProjectFromImageToWorld(Vector2i src,Vector3f &dst,float depth,Matrix4f intrinsic,Matrix4f external)
{
	Vector4f img_homo(depth*src[0],depth*src[1],depth,1.0);
	Vector4f world_homo = external.inverse()*intrinsic.inverse()*img_homo;
	dst[0] = world_homo[0]/world_homo[3];
	dst[1] = world_homo[1]/world_homo[3];
	dst[2] = world_homo[2]/world_homo[3];
}

void createMesh(vector<vector<float>>depthdata,Matrix4f intrinsic,vector<Matrix4f,aligned_allocator<Matrix4f>>external,vector<TriMesh>&mesh)
{
	cout<<"Creating mesh..."<<endl;
	for(int curframe = 0;curframe<depthdata.size();curframe++)
	{
		TriMesh curmesh;
		curmesh.request_vertex_texcoords3D();
		TriMesh::VertexHandle vhandle[IMGWIDTH*IMGHEIGHT];

		//vertex
		for(int y=0;y<IMGHEIGHT;y++)
		{
			for(int x=0;x<IMGWIDTH;x++)
			{
				Vector3f curworldpt;
				ProjectFromImageToWorld(Vector2i(x,y),curworldpt,depthdata[curframe][y*IMGWIDTH+x],intrinsic,external[curframe]);
				vhandle[y*IMGWIDTH+x] = curmesh.add_vertex(TriMesh::Point(curworldpt[0],curworldpt[1],curworldpt[2]));
			}
		}

		//face
		vector<TriMesh::VertexHandle> face_vhandles;
		for(int y=0;y<IMGHEIGHT-1;y++)
		{
			for(int x=0;x<IMGWIDTH-1;x++)
			{
				//anticlockwise
				int v1 = y*IMGWIDTH+x;
				int v2 = (y+1)*IMGWIDTH+x;
				int v3 = (y+1)*IMGWIDTH+x+1;
				int v4 = y*IMGWIDTH+x+1;
				
				//There are holes in the depthmap, we have to avoid these holes
				int validcount = (int)(depthdata[curframe][v1]>0) + (int)(depthdata[curframe][v2]>0) + (int)(depthdata[curframe][v3]>0) + (int)(depthdata[curframe][v4]>0);
				if(validcount == 4)
				{
					face_vhandles.clear();
					face_vhandles.push_back(vhandle[v1]);
					face_vhandles.push_back(vhandle[v2]);
					face_vhandles.push_back(vhandle[v3]);
					curmesh.add_face(face_vhandles);

					face_vhandles.clear();
					face_vhandles.push_back(vhandle[v3]);
					face_vhandles.push_back(vhandle[v4]);
					face_vhandles.push_back(vhandle[v1]);
					curmesh.add_face(face_vhandles);
				}
				if(validcount == 3)
				{
					if(depthdata[curframe][v1] == 0)
					{
						face_vhandles.clear();
						face_vhandles.push_back(vhandle[v2]);
						face_vhandles.push_back(vhandle[v3]);
						face_vhandles.push_back(vhandle[v4]);
						curmesh.add_face(face_vhandles);
					}
					if(depthdata[curframe][v2] == 0)
					{
						face_vhandles.clear();
						face_vhandles.push_back(vhandle[v1]);
						face_vhandles.push_back(vhandle[v3]);
						face_vhandles.push_back(vhandle[v4]);
						curmesh.add_face(face_vhandles);
					}
					if(depthdata[curframe][v3] == 0)
					{
						face_vhandles.clear();
						face_vhandles.push_back(vhandle[v1]);
						face_vhandles.push_back(vhandle[v2]);
						face_vhandles.push_back(vhandle[v4]);
						curmesh.add_face(face_vhandles);
					}
					if(depthdata[curframe][v4] == 0)
					{
						face_vhandles.clear();
						face_vhandles.push_back(vhandle[v1]);
						face_vhandles.push_back(vhandle[v2]);
						face_vhandles.push_back(vhandle[v3]);
						curmesh.add_face(face_vhandles);
					}
				}
			}
		}

		mesh.push_back(curmesh);
	}

}

void computeTextureCoordiante(vector<TriMesh>&mesh,Matrix4f intrinsic,vector<Matrix4f,aligned_allocator<Matrix4f>>external)
{
	for(int i=0;i<mesh.size();i++)
	{
		for(TriMesh::VertexIter v_it = mesh[i].vertices_begin();v_it!=mesh[i].vertices_end();++v_it)
		{
			TriMesh::Point curpt = mesh[i].point(v_it);
			Vector2i imgPt;
			ProjectFromWorldToImage(Vector3f(curpt[0],curpt[1],curpt[2]),imgPt,intrinsic,external[i]);
			if(isValid(imgPt,IMGWIDTH,IMGHEIGHT))
			{
				TriMesh::TexCoord3D tex= TriMesh::TexCoord3D(imgPt[0],imgPt[1],i);
				mesh[i].set_texcoord3D(v_it,tex);
			}
		}

	}
}