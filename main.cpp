#define DEPTHWIDTH 640
#define DEPTHHEIGHT 480
#define COLORWIDTH 640
#define COLORHEIGHT 480
#define FRAMEINTERVAL 10
#define TOTALNUM 55
#define INTERPOLATECOUNT 50
#define DYNNUM 0
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
#pragma comment (lib,"opencv_core248d.lib")
#pragma comment (lib,"opencv_highgui248d.lib")
#pragma comment (lib,"opencv_imgproc248d.lib")
#pragma comment (lib,"OpenMeshCored.lib")
#else
#pragma comment (lib,"opencv_core248.lib")
#pragma comment (lib,"opencv_highgui248.lib")
#pragma comment (lib,"opencv_imgproc248.lib")
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

bool isWriting = true;

//depth data
vector<vector<float>>depthdata;

//camera parameter
vector<Matrix4f,aligned_allocator<Matrix4f>> external_array;
Matrix4f intrinsic_depth;
Matrix4f intrinsic_RGB;
Matrix4f external_depth_to_RGB;
Matrix4f dynmask_external;

//light and material
GLfloat mat_specular[] = {1.0,1.0,1.0,1.0};
GLfloat mat_shininess[] = {100.0};
GLfloat light_position[] = {1.0,1.0,1.0,0.0};
GLfloat light_position2[] = {0.0,0.0,0.0,0.0};
GLfloat light_position3[] = {1.0,-1.0,0.0,0.0};
GLfloat white_light[] = {1.0,1.0,1.0,1.0};

//texture images
vector<Mat>textureimg;
vector<Mat>saveimg;
Mat dynmask;

//Mesh
vector<TriMesh>mesh;
TriMesh largeMesh;	// !!only for debugging
vector<vector<Vector2i>>texturecoordinate;
vector<int> facenumber;

//for navigation
float curlookat[9] = {};
float lookattable[TOTALNUM][9] = {};
float difflookat[9] = {};
static int currentframe = 0;
static int nextframe = 1;
int dynaframe = 0;

static float weight_current = 1.0;

//GL matrix
static int mat_viewport[4] = {};
static float mat_project[16] = {};
static float mat_modelview[16] = {};

//texture
GLuint texture;
//buffer
enum {renderBuffer0,renderbuffer1,saveBuffer,frameBufferNum};
enum {face,vertex,normal,texcoord,bufferObjectNum};
GLuint fbo[frameBufferNum];
GLuint bo[bufferObjectNum];

void OpenGLShow(int argc, char** argv);
void updateTexture(int viewid);
void getvideoTexture(int viewid);
void getColor(int viewid);  //!!only for debugging
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
	sprintf(prefix,"D:/yanhang/RenderProject/room/data6.3");
	readTexture(prefix,TOTALNUM,DYNNUM,FRAMEINTERVAL,textureimg,dynmask);

	readCamera(prefix,TOTALNUM,FRAMEINTERVAL,external_array,dynmask_external);

	intrinsic_depth<<575,0.0,319.5,0.0,0.0,575,239.5,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0;
	intrinsic_RGB<<575,0.0,319.5,0.0,0.0,575,239.5,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0;
	//intrinsic_RGB<<1075.433,0.0,639.5,0.0,0.0,1075.433,479.5,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0;

	/*external_depth_to_RGB<<1.0,0.0,0.0,-0.025,
							0.0,1.0,0.0,0.02,
							0.0,0.0,1.0,-0.00218,
							0.0,0.0,0.0,1.0;*/
	external_depth_to_RGB<<1.0,0.0,0.0,0.0,
							0.0,1.0,0.0,0.0,
							0.0,0.0,1.0,0.0,
							0.0,0.0,0.0,1.0;

	readDepth(prefix,TOTALNUM,FRAMEINTERVAL,depthdata);
	createMesh(depthdata,intrinsic_depth,external_array,mesh,0.4,5.0,facenumber);

	computeTextureCoordiante(mesh,intrinsic_RGB,external_array,external_depth_to_RGB);


	/*cout<<"Saving color..."<<endl;
	for(int i=0;i<mesh.size();i++)
	{
		//getvideoTexture(i);
		getColor(i);
	}*/

	for(int i=0;i<depthdata.size();i++)
	{
		for(TriMesh::VertexIter v_it = mesh[i].vertices_begin();v_it!=mesh[i].vertices_end();++v_it)
			largeMesh.add_vertex(mesh[i].point(v_it));
	}
	char buffer[100];
	sprintf(buffer,"%s/pointcloud/large.off",prefix);
	OpenMesh::IO::write_mesh(largeMesh,string(buffer));


	OpenGLShow(argc,argv);
	return 0;
}

void OpenGLShow(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DEPTH|GLUT_DOUBLE|GLUT_RGBA);
	glutInitWindowSize(DEPTHWIDTH,DEPTHHEIGHT);
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
	glViewport(0,0,DEPTHWIDTH,DEPTHHEIGHT);
	glGetIntegerv(GL_VIEWPORT,mat_viewport);

	for(int i=0;i<depthdata.size();i++)
	{
		Vector3f frame_center;

		Vector4f updir_homo(0.0,-1.0,0.0,1.0);
		Vector4f upstd_homo(0.0,-1.0,0.0,1.0);

		ProjectFromImageToWorld(Vector2i(DEPTHWIDTH/2,DEPTHHEIGHT/2),frame_center,1.0,intrinsic_depth,external_array[i]);

		updir_homo = external_array[i].inverse()*upstd_homo;

		for(int j=0;j<3;j++)
		{
			Matrix4f curext = external_array[i].inverse();
			lookattable[i][j] = curext(j,3);
		}
		for(int j=3;j<6;j++)
			lookattable[i][j] = frame_center(j-3);
		for(int j=6;j<9;j++)
			lookattable[i][j] = updir_homo(j-6);
	}

	for(int i=0;i<9;i++)
		curlookat[i] = lookattable[currentframe][i];

	for(int i=0;i<9;i++)
		difflookat[i] = (lookattable[nextframe][i] - lookattable[currentframe][i])/static_cast<float>(INTERPOLATECOUNT);

	updateview();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(35.0,(float)DEPTHWIDTH/(float)DEPTHHEIGHT,0.001,5.0);

	//Buffer Objects
	glGenBuffers(bufferObjectNum,bo);
	//face
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,bo[face]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,3*(DEPTHHEIGHT-1)*(DEPTHWIDTH-1)*2*sizeof(GLuint),NULL,GL_DYNAMIC_DRAW);
	GLuint *glface = (GLuint*)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER,GL_WRITE_ONLY);
	for(int i=0;i<(DEPTHWIDTH-1)*(DEPTHHEIGHT-1)*2*3;i++) glface[i] = i;
	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);

	//vertex position
	GLuint loc;
	glBindBuffer(GL_ARRAY_BUFFER,bo[vertex]);
	glBufferData(GL_ARRAY_BUFFER,9*(DEPTHHEIGHT-1)*(DEPTHWIDTH-1)*2*sizeof(GLfloat),NULL,GL_DYNAMIC_DRAW);
	loc = glGetAttribLocation(_shader.GetProgramID(),"position");
	glEnableVertexAttribArray(loc);
	glVertexAttribPointer(loc,3,GL_FLOAT,0,0,BUFFER_OFFSET(0));
	//texture coordinate
	glBindBuffer(GL_ARRAY_BUFFER,bo[texcoord]);
	glBufferData(GL_ARRAY_BUFFER,9*(DEPTHHEIGHT-1)*(DEPTHWIDTH-1)*2*sizeof(GLfloat),NULL,GL_DYNAMIC_DRAW);
	loc = glGetAttribLocation(_shader.GetProgramID(),"texcoord");
	glEnableVertexAttribArray(loc);
	glVertexAttribPointer(loc,3,GL_FLOAT,0,0,BUFFER_OFFSET(0));

	//normals
	glBindBuffer(GL_ARRAY_BUFFER,bo[normal]);
	glBufferData(GL_ARRAY_BUFFER,9*(DEPTHHEIGHT-1)*(DEPTHWIDTH-1)*2*sizeof(GLfloat),NULL,GL_DYNAMIC_DRAW);

	//set light
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT_AND_BACK,GL_SHININESS,mat_shininess);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, white_light);
    glLightfv(GL_LIGHT0, GL_SPECULAR, white_light);
    
    glLightfv(GL_LIGHT1, GL_POSITION, light_position2);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, white_light);
    glLightfv(GL_LIGHT1, GL_SPECULAR, white_light);
    
    glLightfv(GL_LIGHT2, GL_POSITION, light_position3);
    glLightfv(GL_LIGHT2, GL_DIFFUSE, white_light);
    glLightfv(GL_LIGHT2, GL_SPECULAR, white_light);
    
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHT1);
    glEnable(GL_LIGHT2);

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
	glTexImage3D(GL_TEXTURE_2D_ARRAY,0,GL_RGB,COLORWIDTH,COLORHEIGHT,textureimg.size(),0,GL_BGR,GL_UNSIGNED_BYTE,NULL);
	for(int i=0;i<textureimg.size();i++)
		glTexSubImage3D(GL_TEXTURE_2D_ARRAY,0,0,0,i,textureimg[i].cols,textureimg[i].rows,1,GL_BGR,GL_UNSIGNED_BYTE,textureimg[i].data);

	//set callback functions
	glutKeyboardFunc(ProcessNormalKeys);
	glutSpecialFunc(ProcessSpecialKeys);
	glutDisplayFunc(render);
	glutIdleFunc(idlefunction);
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

	glShadeModel(GL_SMOOTH);

	/*_shader.EnableShader();

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
	glUniform1i(loc,textureimg.size());*/

	glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);

	//current frame
	data2GPU(currentframe);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,bo[face]);
	glDrawElements(GL_TRIANGLES,facenumber[currentframe]*3,GL_UNSIGNED_INT,0);

	//next frame

	/*loc = glGetUniformLocation(_shader.GetProgramID(),"weight");
	if(loc == -1)
	{
		cout<<"Cannot find weight"<<endl;
		system("pause");
	}
	glUniform1f(loc,1.0-weight_current);*/

	/*glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	glBlendEquation(GL_FUNC_ADD);
	glEnable(GL_BLEND);

	mat_specular[3] = (GLfloat)(1.0-weight_current);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
	data2GPU(nextframe);
	glDrawElements(GL_TRIANGLES,facenumber[nextframe]*3,GL_UNSIGNED_INT,0);

	glDisable(GL_BLEND);*/

	//_shader.DisableShader();

	//save the image
	if(isWriting && saveimg.size()<1000)
	{
		glReadBuffer(GL_BACK);
		Mat curimg = Mat(DEPTHHEIGHT,DEPTHWIDTH,CV_8UC3);
		glReadPixels(0,0,DEPTHWIDTH,DEPTHHEIGHT,GL_BGR,GL_UNSIGNED_BYTE,curimg.data);
		flip(curimg,curimg,0);
		saveimg.push_back(curimg);
	}

	glFlush();
	glutSwapBuffers();
}

void getColor(int viewid)
{
	for(TriMesh::VertexIter v_it = mesh[viewid].vertices_begin();v_it!=mesh[viewid].vertices_end();++v_it)
	{
		TriMesh::Point curpt = mesh[viewid].point(v_it);
		Vector2i curtex;
		ProjectFromWorldToImage(Vector3f(curpt[0],curpt[1],curpt[2]),curtex,intrinsic_RGB,external_depth_to_RGB*external_array[viewid]);
		if(isValid(curtex,COLORWIDTH,COLORHEIGHT))
		{
			Vec3b curPix = textureimg[viewid+DYNNUM].at<Vec3b>(curtex[1],curtex[0]);
			TriMesh::Color curcolor = TriMesh::Color(curPix[2],curPix[1],curPix[0]);
			mesh[viewid].set_color(v_it,curcolor);
		}
	}

	char colorpath[100];
	sprintf(colorpath,"%s/pointcloud/colorcloud%03d.off",prefix,viewid);
	OpenMesh::IO::Options wopt;
	wopt += OpenMesh::IO::Options::VertexColor;
	OpenMesh::IO::write_mesh(mesh[viewid],string(colorpath),wopt);

}

void idlefunction()
{
	if(dynaframe == DYNNUM)
		dynaframe = 0;
	else
		dynaframe++;

	getvideoTexture(currentframe);

	getvideoTexture(nextframe);

	render();
}

void getvideoTexture(int viewid)
{
	for(TriMesh::VertexIter v_it = mesh[viewid].vertices_begin();v_it!=mesh[viewid].vertices_end();++v_it)
	{
		TriMesh::Point curpt = mesh[viewid].point(v_it);
		Vector2i imgPt;
		ProjectFromWorldToImage(Vector3f(curpt[0],curpt[1],curpt[2]),imgPt,intrinsic_RGB,external_depth_to_RGB*dynmask_external);
		if(isValid(imgPt,COLORWIDTH,COLORHEIGHT))
		{
			int locmask = static_cast<int>(dynmask.at<uchar>(imgPt[1],imgPt[0]));
			if(locmask > 200)
			{
				TriMesh::TexCoord3D curtex = TriMesh::TexCoord3D(imgPt[0],imgPt[1],dynaframe);
				mesh[viewid].set_texcoord3D(v_it,curtex);
			}
		}
	}
}

void updateview()
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt((GLdouble)curlookat[0], (GLdouble)curlookat[1], (GLdouble)curlookat[2], (GLdouble)curlookat[3], (GLdouble)curlookat[4], (GLdouble)curlookat[5],(GLdouble)curlookat[6], (GLdouble)curlookat[7],(GLdouble)curlookat[8]);
}

void data2GPU(int viewid)
{
	int ind = 0;
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
	glVertexPointer(3,GL_FLOAT,0,0);

	//normals
	ind = 0;
    glBindBuffer(GL_ARRAY_BUFFER,bo[normal]);
    GLfloat* glNorMal = (GLfloat*)glMapBuffer(GL_ARRAY_BUFFER,GL_WRITE_ONLY);
    for(TriMesh::FaceIter f_it = mesh[viewid].faces_begin();f_it!=mesh[viewid].faces_end();++f_it)
	{
		for(TriMesh::FaceVertexIter fv_it = mesh[viewid].fv_iter(f_it);fv_it;++fv_it)
		{
			TriMesh::Normal curnormal = mesh[viewid].normal(fv_it);
			glNorMal[ind*3+0] = (GLfloat)curnormal[0];
			glNorMal[ind*3+1] = (GLfloat)curnormal[1];
			glNorMal[ind*3+2] = (GLfloat)curnormal[2];
			ind++;
		}
	}
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glNormalPointer(GL_FLOAT, 0, 0);

	//texture
	//updateTexture(viewid);

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
			gltexcoord[ind*3+0] = (GLfloat)(static_cast<float>(curtex[0])/static_cast<float>(COLORWIDTH));
			gltexcoord[ind*3+1] = (GLfloat)(static_cast<float>(curtex[1])/static_cast<float>(COLORHEIGHT));
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
	if(key == 's' || key=='S')
	{
		cout<<"Saving..."<<endl;
		for(int i=0;i<saveimg.size();i++)
		{
			char buffer[100];
			sprintf(buffer,"%s/save/frame%03d.png",prefix,i);
			imwrite(buffer,saveimg[i]);
		}
		cout<<"Save complete!"<<endl;
	}
}

void ProcessSpecialKeys(int key,int x,int y)
{
	if(key == GLUT_KEY_RIGHT)
	{
		cout<<"currentframe: "<<currentframe<<endl;

		for(int i=0;i<INTERPOLATECOUNT;i++)
		{
			for(int j=0;j<9;j++)
				curlookat[j] += difflookat[j];
			updateview();
			weight_current -= 1.0/static_cast<float>(INTERPOLATECOUNT);
			render();
		}

		currentframe = nextframe;
		nextframe++;

		cout<<"current lookat:"<<endl;
		for(int i=0;i<9;i++)
			cout<<curlookat[i]<<' ';
		cout<<endl;
		cout<<"lookattable "<<currentframe<<endl;
		for(int i=0;i<9;i++)
			cout<<lookattable[currentframe][i]<<' ';
		cout<<endl;

		if(nextframe == depthdata.size())
			nextframe = 0;

		for(int i=0;i<9;i++)
			difflookat[i] = (lookattable[nextframe][i]-lookattable[currentframe][i])/static_cast<float>(INTERPOLATECOUNT);

		for(int i=0;i<9;i++)
			curlookat[i] = lookattable[currentframe][i];
		weight_current = 1.0;
		updateview();
		render();
	}
	if(key == GLUT_KEY_LEFT)
	{
		nextframe = currentframe;
		if(currentframe == 0)
			currentframe = depthdata.size()-1;
		else
			currentframe--;
		for(int i=0;i<9;i++)
			difflookat[i] = (lookattable[nextframe][i]-lookattable[currentframe][i])/static_cast<float>(INTERPOLATECOUNT);

		for(int i=0;i<9;i++)
			curlookat[i] = lookattable[currentframe][i];
		updateview();
		glutPostRedisplay();
	}
	if(key == GLUT_KEY_UP)
	{
		curlookat[4] += 0.05;
		updateview();
		glutPostRedisplay();
	}
	if(key == GLUT_KEY_DOWN)
	{
		curlookat[4] -= 0.05;
		updateview();
		glutPostRedisplay();
	}
}