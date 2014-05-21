#define IMGWIDTH 640
#define IMGHEIGHT 480
#define FRAMEINTERVAL 5
#define TOTALNUM 62
#define INTERPOLATECOUNT 10
#define BUFFER_OFFSET(i) ((uchar*)NULL + (i))

#include <iostream>
#include <fstream>
#include <glew\include\GL\glew.h>
#include <freeglut\include\GL\freeglut.h>
#include <opencv2\opencv.hpp>
#include <Eigen\Eigen>
#include "GLShader.h"
#include "utilities.h"
#include <math.h>
#if _DEBUG
#pragma comment (lib,"opencv_core245d.lib")
#pragma comment (lib,"opencv_highgui245d.lib")
#pragma comment (lib,"opencv_imgproc245d.lib")
#pragma comment (lib,"OpenMeshCored.lib")
#else
#pragma comment (lib,"opencv_core245.lib")
#pragma comment (lib,"opencv_highgui245.lib")
#pragma comment (lib,"opencv_imgproc245.lib")
#pragma comment (lib,"OpenMeshCore.lib")
#endif

#pragma comment (lib, "freeglut.lib")
#pragma comment (lib, "glew32.lib")

using namespace std;
using namespace cv;
using namespace Eigen;

static int viewcount = TOTALNUM/FRAMEINTERVAL;

Shader _shader;
//path
char prefix[100];

//depth data
vector<vector<float>>depthdata;

//camera parameter
vector<Matrix4f,aligned_allocator<Matrix4f>> external_array;
Matrix4f intrinsic;

//texture images
vector<Mat>textureimg;

//Mesh
vector<TriMesh>mesh;
TriMesh largeMesh;
vector<vector<Vector2i>>texturecoordinate;

//for navigation
float curlookat[6] = {};
float lookattable[TOTALNUM][6] = {};
float difflookat[6] = {};
static int currentframe = 0;
static int nextframe = 1;

static float weight_current = 1.0;

//GL matrix
static int mat_viewport[4] = {};
static float mat_project[16] = {};
static float mat_modelview[16] = {};

//texture
GLuint texture;
//buffer
enum {renderBuffer0,renderbuffer1,saveBuffer,frameBufferNum};
enum {face,vertex,texcoord,bufferObjectNum};
GLuint fbo[frameBufferNum];
GLuint bo[bufferObjectNum];

void OpenGLShow(int argc, char** argv);
void updateTexture(int viewid);
void updateview();

//Glut callback functions
void ProcessNormalKeys(unsigned char key, int x,int y);
void ProcessSpecialKeys(int key,int x,int y);
void render();
void idlefunction();
void close();
void data2GPU(int viewid);

int main(int argc,char**argv)
{
	sprintf(prefix,"D:/yanhang/RenderProject/room/data5.18");
	readTexture(prefix,TOTALNUM,FRAMEINTERVAL,textureimg);

	readCamera(prefix,TOTALNUM,FRAMEINTERVAL,external_array);
	intrinsic<<575,0.0,319.5,0.0,0.0,575,239.5,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0;
	readDepth(prefix,TOTALNUM,FRAMEINTERVAL,depthdata);
	createMesh(depthdata,intrinsic,external_array,mesh);
	computeTextureCoordiante(mesh,intrinsic,external_array);

	cout<<"texture num:"<<textureimg.size()<<endl;
	cout<<"depth num:"<<depthdata.size()<<endl;
	cout<<"camera num:"<<external_array.size()<<endl;
	viewcount = external_array.size();

	/*for(int i=0;i<mesh.size();i++)
	{
		char buffer[100];
		sprintf(buffer,"%s/pointcloud/frame%03d.off",prefix,i);
		OpenMesh::IO::write_mesh(mesh[i],string(buffer));
	}*/

	/*for(int i=0;i<mesh.size();i++)
	{
		for(TriMesh::VertexIter v_it = mesh[i].vertices_begin();v_it!=mesh[i].vertices_end();++v_it)
			largeMesh.add_vertex(mesh[i].point(v_it));
	}
	char buffer[100];
	sprintf(buffer,"%s/pointcloud/large.off",prefix);
	OpenMesh::IO::write_mesh(largeMesh,string(buffer));*/


	OpenGLShow(argc,argv);
	return 0;
}

void OpenGLShow(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DEPTH|GLUT_DOUBLE|GLUT_RGBA);
	glutInitWindowSize(IMGWIDTH,IMGHEIGHT);
	glutInitWindowPosition(100,100);
	glutCreateWindow("View-dependent & time-dependent Rendering");

	if(glewInit() != GLEW_OK)
	{
		cout<<"Glew failed to initialize!"<<endl;
		system("pause");
		exit(-1);
	}

	
	if(!GLEW_ARB_vertex_shader || !GLEW_ARB_fragment_shader)
	{
		cout<<"GLSL not supported!"<<endl;
		system("pause");
		exit(-1);
	}
	_shader.ReadVertextShader("texture_array.vert");
	_shader.ReadFragmentShader("texture_array.frag");
	_shader.SetShader();

	glEnable(GL_DEPTH);
	glClearColor(0,0,0,0);
	glClearDepth(1.0);
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

	//set camera
	glViewport(0,0,IMGWIDTH,IMGHEIGHT);
	glGetIntegerv(GL_VIEWPORT,mat_viewport);

	for(int i=0;i<depthdata.size();i++)
	{
		Vector3f frame_center;
		ProjectFromImageToWorld(Vector2i(IMGWIDTH/2,IMGHEIGHT/2),frame_center,depthdata[i][IMGWIDTH*IMGHEIGHT/2],intrinsic,external_array[i]);
		for(int j=0;j<3;j++)
			lookattable[i][j] = external_array[i](j,3);
		for(int j=3;j<6;j++)
			lookattable[i][j] = frame_center(j-3) - external_array[i](j-3,3);
	}

	cout<<"Look at table:"<<endl;
	for(int i=0;i<depthdata.size();i++)
	{
		for(int j=0;j<6;j++)
			cout<<lookattable[i][j]<<' ';
		cout<<endl;
	}

	for(int i=0;i<6;i++)
		curlookat[i] = lookattable[currentframe][i];

	for(int i=0;i<6;i++)
		difflookat[i] = (lookattable[nextframe][i] - lookattable[currentframe][i])/static_cast<float>(INTERPOLATECOUNT);

	updateview();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(30.0,(float)IMGWIDTH/(float)IMGHEIGHT,0.001,5.0);

	//Buffer Objects
	glGenBuffers(bufferObjectNum,bo);
	//face
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,bo[face]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,3*(IMGHEIGHT-1)*(IMGWIDTH-1)*2*sizeof(GLuint),NULL,GL_DYNAMIC_DRAW);
	//vertex position
	GLuint loc;
	glBindBuffer(GL_ARRAY_BUFFER,bo[vertex]);
	glBufferData(GL_ARRAY_BUFFER,9*(IMGHEIGHT-1)*(IMGWIDTH-1)*2*sizeof(GLfloat),NULL,GL_DYNAMIC_DRAW);
	loc = glGetAttribLocation(_shader.GetProgramID(),"position");
	glEnableVertexAttribArray(loc);
	glVertexAttribPointer(loc,3,GL_FLOAT,0,0,BUFFER_OFFSET(0));
	//texture coordinate
	glBindBuffer(GL_ARRAY_BUFFER,bo[texcoord]);
	glBufferData(GL_ARRAY_BUFFER,9*(IMGHEIGHT-1)*(IMGWIDTH-1)*2*sizeof(GLfloat),NULL,GL_DYNAMIC_DRAW);
	loc = glGetAttribLocation(_shader.GetProgramID(),"texcoord");
	glEnableVertexAttribArray(loc);
	glVertexAttribPointer(loc,3,GL_FLOAT,0,0,BUFFER_OFFSET(0));

	//load textures
	glActiveTexture(GL_TEXTURE0);
	glEnable(GL_TEXTURE_3D);
	glGenTextures(1,&texture);
	glBindTexture(GL_TEXTURE_2D_ARRAY,texture);
	glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_WRAP_S,GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_WRAP_T,GL_REPEAT);
	glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_MODULATE);
	glTexImage3D(GL_TEXTURE_2D_ARRAY,0,GL_RGB,IMGWIDTH,IMGHEIGHT,textureimg.size(),0,GL_BGR,GL_UNSIGNED_BYTE,NULL);
	for(int i=0;i<textureimg.size();i++)
		glTexSubImage3D(GL_TEXTURE_2D_ARRAY,0,0,0,i,textureimg[i].cols,textureimg[i].rows,1,GL_BGR,GL_UNSIGNED_BYTE,textureimg[i].data);

	

	data2GPU(currentframe);

	//set callback functions
	glutKeyboardFunc(ProcessNormalKeys);
	glutSpecialFunc(ProcessSpecialKeys);
	glutDisplayFunc(render);
	//glutIdleFunc(idlefunction);
	glutCloseFunc(close);
	glutMainLoop();

	glDeleteBuffers(bufferObjectNum,bo);
	glDeleteTextures(1,&texture);
}

//render with current frame and next frame
void render()
{
	glClearColor(0.0,0.0,0.0,1.0);
	glClear(GL_COLOR_BUFFER_BIT);

	_shader.EnableShader();

	//uploat modelview and projection matrix
	GLuint loc;
	glGetFloatv(GL_MODELVIEW_MATRIX,mat_modelview);
	loc = glGetUniformLocation(_shader.GetProgramID(),"mv_mat");
	if(loc == -1)
	{
		cout<<"Cannot find mv_mat"<<endl;
		system("pause");
	}
	glUniformMatrix4fv(loc,1,false,mat_modelview);

	glGetFloatv(GL_PROJECTION_MATRIX,mat_project);
	loc = glGetUniformLocation(_shader.GetProgramID(),"mp_mat");
	if(loc == -1)
	{
		cout<<"Cannot find mp_mat"<<endl;
		system("pause");
	}
	glUniformMatrix4fv(loc,1,false,mat_project);

	loc = glGetUniformLocation(_shader.GetProgramID(),"weight");
	if(loc == -1)
	{
		cout<<"Cannot find weight"<<endl;
		system("pause");
	}
	glUniform1f(loc,weight_current);

	//upload texture
	loc = glGetUniformLocation(_shader.GetProgramID(),"tex_array");
	if(loc == -1)
	{
		cout<<"Cannot find tex_array"<<endl;
		system("pause");
	}
	glUniform1i(loc,0);

	loc = glGetUniformLocation(_shader.GetProgramID(),"TexNum");
	if(loc == -1)
	{
		cout<<"Cannot find TexNum"<<endl;
		system("pause");
	}
	glUniform1i(loc,textureimg.size());

	//current frame
	//data2GPU(currentframe);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,bo[face]);
	glDrawElements(GL_TRIANGLES,(IMGHEIGHT-1)*(IMGWIDTH-1)*2*3,GL_UNSIGNED_INT,0);

	/*//next frame

	loc = glGetUniformLocation(_shader.GetProgramID(),"weight");
	if(loc == -1)
	{
		cout<<"Cannot find weight"<<endl;
		system("pause");
	}
	glUniform1f(loc,1.0-weight_current);
	glBlendFunc(GL_SRC_ALPHA,GL_DST_ALPHA);
	glBlendEquation(GL_FUNC_ADD);

	glEnable(GL_BLEND);
	data2GPU(nextframe);
	glDrawElements(GL_TRIANGLES,(IMGHEIGHT-1)*(IMGWIDTH-1)*2*3,GL_UNSIGNED_INT,0);

	glDisable(GL_BLEND);*/

	_shader.DisableShader();

	glFlush();
	glutSwapBuffers();
}

void idlefunction()
{

}

void updateview()
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt((GLdouble)curlookat[0], -1*(GLdouble)curlookat[1], (GLdouble)curlookat[2], (GLdouble)curlookat[3], (GLdouble)curlookat[4], (GLdouble)curlookat[5], 0.0, -1.0, 0.0);
}

void data2GPU(int viewid)
{
	int ind = 0;
	//faces
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,bo[face]);
	GLuint *glface = (GLuint*)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER,GL_WRITE_ONLY);
	for(int i=0;i<mesh[viewid].n_faces()*3;i++) glface[i] = i;
	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);

	//vertex position
	glBindBuffer(GL_ARRAY_BUFFER,bo[vertex]);
	GLfloat *glposition = (GLfloat*)glMapBuffer(GL_ARRAY_BUFFER,GL_WRITE_ONLY);
	ind = 0;
	for(TriMesh::FaceIter f_it = mesh[viewid].faces_begin();f_it!=mesh[viewid].faces_end();++f_it)
	{
		for(TriMesh::FaceVertexIter fv_it = mesh[viewid].fv_iter(f_it);fv_it;++fv_it)
		{
			TriMesh::Point curpt = mesh[viewid].point(fv_it);
			glposition[ind*3+0] = (GLfloat)curpt[0];
			glposition[ind*3+1] = (GLfloat)curpt[1];
			glposition[ind*3+2] = (GLfloat)curpt[2];
			ind++;
		}
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);

	//texture
	updateTexture(viewid);

}

void updateTexture(int viewid)
{
	int ind = 0;
	glBindBuffer(GL_ARRAY_BUFFER,bo[texcoord]);
	GLfloat* gltexcoord = (GLfloat*)glMapBuffer(GL_ARRAY_BUFFER,GL_WRITE_ONLY);
	for(TriMesh::FaceIter f_it = mesh[viewid].faces_begin();f_it!=mesh[viewid].faces_end();++f_it)
	{
		for(TriMesh::FaceVertexIter fv_it = mesh[viewid].fv_iter(f_it);fv_it;++fv_it)
		{
			TriMesh::TexCoord3D curtex = mesh[viewid].texcoord3D(fv_it);
			gltexcoord[ind*3+0] = (GLfloat)(static_cast<float>(curtex[0])/static_cast<float>(IMGWIDTH));
			gltexcoord[ind*3+1] = (GLfloat)(static_cast<float>(curtex[1])/static_cast<float>(IMGHEIGHT));
			gltexcoord[ind*3+2] = (GLfloat)curtex[2];
			ind++;
		}
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

//callback functions
void close()
{
	_shader.deleteshader();
}

void ProcessNormalKeys(unsigned char key,int x,int y)
{
}

void ProcessSpecialKeys(int key,int x,int y)
{
	if(key == GLUT_KEY_RIGHT)
	{
		cout<<"currentframe: "<<currentframe<<endl;

		for(int i=0;i<INTERPOLATECOUNT-1;i++)
		{
			for(int j=0;j<6;j++)
				curlookat[j] += difflookat[j];
			updateview();
			weight_current -= 1.0/static_cast<float>(INTERPOLATECOUNT);
			//cout<<weight_current<<endl;
			render();
		}

		currentframe = nextframe;
		nextframe++;

		if(nextframe == depthdata.size())
			nextframe = 0;

		for(int i=0;i<6;i++)
			difflookat[i] = (lookattable[nextframe][i]-lookattable[currentframe][i])/static_cast<float>(INTERPOLATECOUNT);

		for(int i=0;i<6;i++)
			curlookat[i] = lookattable[currentframe][i];
		weight_current = 1.0;
		data2GPU(currentframe);
		updateview();
		//render();

		glutPostRedisplay();
	}
	if(key == GLUT_KEY_LEFT)
	{
		curlookat[1] -= 0.1;
		updateview();
		glutPostRedisplay();
	}
}