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

void readTexture(char *path,int num,int dynnum,int frameinterval,vector<Mat>&textureimg,Mat &mask)
{
	cout<<"reading texture..."<<endl;
	/*for(int i=0;i<dynnum;i++)
	{
		char dynbuffer[100];
		sprintf(dynbuffer,"%s/image/dynarea%03d.png",path,i);
		Mat curdyn = imread(dynbuffer);
		flip(curdyn,curdyn,1);
		textureimg.push_back(curdyn);
	}*/

	for(int i=0;i<num;i=i+frameinterval)
	{
		char buffer[100];
		sprintf(buffer,"%s/image/frame%03d_2.png",path,i);
		Mat curimg = imread(string(buffer));
		flip(curimg,curimg,1);
		textureimg.push_back(curimg);
	}
	/*char buffer[100];
	sprintf(buffer,"%s/image/frame_reference.png",path);
	mask = imread(string(buffer));
	cvtColor(mask,mask,CV_RGB2GRAY);
	flip(mask,mask,1);*/

}

void readCamera(char *path,int num,int frameinterval,vector<Matrix4f,aligned_allocator<Matrix4f>>&external,Matrix4f &dynmask_external)
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
		fin.close();
	}
	
	/*char buffer[100];
	sprintf(buffer,"%s/Geometry/frame_reference.txt",path);
	ifstream maskfin(buffer);
	for(int y=0;y<4;y++)
	{
		for(int x=0;x<4;x++)
			maskfin>>dynmask_external(y,x);
	}
	dynmask_external(3,3) = 1.0;
	dynmask_external.transposeInPlace();*/
}

void readDepth(char *path,int num,int frameinterval,vector<vector<float>>&depthdata)
{
	cout<<"reading depth..."<<endl;
	for(int i=0;i<num;i=i+frameinterval)
	{
		cout<<i<<' ';
		char buffer[100];
		sprintf(buffer,"%s/depth/frame_depth%03d.txt",path,i);
		ifstream fin(buffer);
		vector<float>curdepth;
		float temp;
		for(int i=0;i<DEPTHWIDTH*DEPTHHEIGHT;i++)
		{
			fin>>temp;
			curdepth.push_back(temp);
		}
		fin.close();
		depthdata.push_back(curdepth);
	}
	cout<<endl;
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

void createMesh(vector<vector<float>>depthdata,Matrix4f intrinsic,vector<Matrix4f,aligned_allocator<Matrix4f>>external,vector<TriMesh>&mesh,float mindistance,float maxdistance,vector<int>&facenumber)
{
	cout<<"Creating mesh..."<<endl;
	for(int curframe = 0;curframe<depthdata.size();curframe++)
	{
		TriMesh curmesh;
		curmesh.request_vertex_texcoords3D();
		curmesh.request_vertex_colors();
		TriMesh::VertexHandle vhandle[DEPTHWIDTH*DEPTHHEIGHT];

		//vertex
		for(int y=0;y<DEPTHHEIGHT;y++)
		{
			for(int x=0;x<DEPTHWIDTH;x++)
			{
				Vector3f curworldpt;
				ProjectFromImageToWorld(Vector2i(x,y),curworldpt,depthdata[curframe][y*DEPTHWIDTH+x],intrinsic,external[curframe]);
				vhandle[y*DEPTHWIDTH+x] = curmesh.add_vertex(TriMesh::Point(curworldpt[0],curworldpt[1],curworldpt[2]));
			}
		}

		//face
		vector<TriMesh::VertexHandle> face_vhandles;
		for(int y=0;y<DEPTHHEIGHT-1;y++)
		{
			for(int x=0;x<DEPTHWIDTH-1;x++)
			{
				//anticlockwise
				int v1 = y*DEPTHWIDTH+x;
				int v2 = (y+1)*DEPTHWIDTH+x;
				int v3 = (y+1)*DEPTHWIDTH+x+1;
				int v4 = y*DEPTHWIDTH+x+1;
				
				//There are holes in the depthmap, we have to avoid these holes
				int validcount = ((depthdata[curframe][v1]>mindistance)&&(depthdata[curframe][v1]<maxdistance)) + ((depthdata[curframe][v2]>mindistance)&&(depthdata[curframe][v2]<maxdistance))
					+ ((depthdata[curframe][v3]>mindistance)&&(depthdata[curframe][v3]<maxdistance)) + ((depthdata[curframe][v4]>mindistance)&&(depthdata[curframe][v4]<maxdistance));
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
					if(depthdata[curframe][v1] <mindistance || depthdata[curframe][v1]>maxdistance)
					{
						face_vhandles.clear();
						face_vhandles.push_back(vhandle[v2]);
						face_vhandles.push_back(vhandle[v3]);
						face_vhandles.push_back(vhandle[v4]);
						curmesh.add_face(face_vhandles);
					}
					if(depthdata[curframe][v2] <mindistance || depthdata[curframe][v2]>maxdistance)
					{
						face_vhandles.clear();
						face_vhandles.push_back(vhandle[v1]);
						face_vhandles.push_back(vhandle[v3]);
						face_vhandles.push_back(vhandle[v4]);
						curmesh.add_face(face_vhandles);
					}
					if(depthdata[curframe][v3] <mindistance || depthdata[curframe][v2]>maxdistance)
					{
						face_vhandles.clear();
						face_vhandles.push_back(vhandle[v1]);
						face_vhandles.push_back(vhandle[v2]);
						face_vhandles.push_back(vhandle[v4]);
						curmesh.add_face(face_vhandles);
					}
					if(depthdata[curframe][v4] <mindistance || depthdata[curframe][v2]>maxdistance)
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
		facenumber.push_back(curmesh.n_faces());
	}

}

void computeTextureCoordiante(vector<TriMesh>&mesh,Matrix4f intrinsic_RGB,vector<Matrix4f,aligned_allocator<Matrix4f>>external,Matrix4f external_depth_to_RGB)
{
	for(int i=0;i<mesh.size();i++)
	{
		for(TriMesh::VertexIter v_it = mesh[i].vertices_begin();v_it!=mesh[i].vertices_end();++v_it)
		{
			TriMesh::Point curpt = mesh[i].point(v_it);
			Vector2i imgPt;

			ProjectFromWorldToImage(Vector3f(curpt[0],curpt[1],curpt[2]),imgPt,intrinsic_RGB,external_depth_to_RGB*external[i]);
			if(isValid(imgPt,COLORWIDTH,COLORHEIGHT))
			{
				TriMesh::TexCoord3D tex= TriMesh::TexCoord3D(imgPt[0],imgPt[1],i+DYNNUM);
				mesh[i].set_texcoord3D(v_it,tex);
			}
		}

	}
}