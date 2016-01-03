
#ifdef _MSC_BUILD
#include "../vs2013/opencv_2.4.9_lib.hpp"
#endif

#include <opencv2/opencv.hpp>
using namespace cv;

#include "SOCP.h"

#include <iostream>
#include <fstream>
#include <cstdio>
using namespace std;


static void testChess(void)
{
	//Size csz(3, 3); Mat image = imread("../../../../data/perspective/800px-Chess_board_opening_staunton.jpg");
	//Size csz(3, 3); Mat image = imread("../../../../data/perspective/800px-Chess_board_with_chess_set_in_opening_position_2012_PD_02.jpg");
	//Size csz(3, 3); Mat image = imread("../../../../data/perspective/800px-Malampuzha_Garden_Chess_Board.JPG");
	//Size csz(3, 3); Mat image = imread("../../../../data/perspective/800px-Wade_whimsies_4o06.jpg");
	//Size csz(3, 3); Mat image = imread("../../../../data/perspective/Heavy_Chess_(4531534307).jpg");
	//Size csz(9, 6); Mat image = imread("../../../../data/perspective/chessboard.jpg");
	Size csz(9, 6); Mat image = imread("../../../../data/perspective/ref.bmp");

	vector<Point2f> corners;
	bool found = findChessboardCorners(image, csz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
	cout << found << endl;
	int h = image.size().height;
	for (vector<Point2f>::iterator it = corners.begin(); it != corners.end(); it++)
	{
		cout << it->x << " " << (h - it->y) << endl;
	}
	drawChessboardCorners(image, csz, corners, found);

	namedWindow("debug");
	imshow("debug", image);

	waitKey(0);
}

static void testHough(void)
{
	Mat image = imread("../../../../data/perspective/ref.bmp");
	//Mat image = imread("../../../../data/perspective/800px-Chess_board_opening_staunton.jpg");
	//Mat image = imread("../../../../data/perspective/800px-Chess_board_with_chess_set_in_opening_position_2012_PD_02.jpg");
	//Mat image = imread("../../../../data/perspective/800px-Malampuzha_Garden_Chess_Board.JPG");
	//Mat image = imread("../../../../data/perspective/800px-Wade_whimsies_4o06.jpg");
	//Mat image = imread("../../../../data/perspective/Heavy_Chess_(4531534307).jpg");
	//Mat image = imread("../../../../data/perspective/chessboard.jpg");
	Mat edges;
	vector<Vec4i> lines;
	double minLineLength = sqrt(image.size().area()) / 64;

	Canny(image, edges, 200, 200, 3);
	HoughLinesP(edges, lines, 1, CV_PI / 180, 80, minLineLength, 10);
	for (size_t i = 0; i < lines.size(); i++)
	{
		line(image, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 3);
	}
	namedWindow("debug");
	imshow("debug", image);

	waitKey(0);
}

static void testSOCP(void)
{
#if 0
	Size csz(9, 6); Mat image = imread("../../../../data/perspective/ref.bmp");

	vector<Point2f> corners;
	bool found = findChessboardCorners(image, csz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
#if 0
	drawChessboardCorners(image, csz, corners, found);
	namedWindow("debug");
	imshow("debug", image);
	waitKey(0);
#endif
	assert(found);

	//

	IPM_Vector *z = new IPM_Vector[csz.area()];
	IPM_Vector *u = new IPM_Vector[csz.area()];
	IPM_Scalar z_0 = 0;
	IPM_Scalar z_1 = IPM_Scalar(csz.height - 1);
	vector<Point2f>::iterator it = corners.begin();

	for (int i = 0; i < csz.area(); i++)
	{
		assert(it != corners.end());

		z[i] = IPM_Vector(4);
		z[i](0) = z_0;
		z[i](1) = z_1;
		z[i](2) = 0;
		z[i](3) = 1;

		u[i] = IPM_Vector(2);
		u[i](0) = (it->x - image.size().width / 2);
		u[i](1) = (image.size().height / 2 - it->y);

#if 0
		cout << z[i].transpose() << "    ";
		cout << u[i].transpose() << endl;
#endif

		if (++z_0 == csz.width)
		{
			z_0 = 0;
			z_1--;
		}
		it++;
	}

	z[0] = z[0];
	u[0] = u[0];
	z[1] = z[8];
	u[1] = u[8];
	z[2] = z[5 * 9];
	u[2] = u[5 * 9];
	z[3] = z[5 * 9 + 8];
	u[3] = u[5 * 9 + 8];
#endif

	Size csz(2, 2);
	Mat image = imread("../../../../data/perspective/443px-Dados_oca_logrono.jpg");

	const IPM_uint c = 8;

	IPM_Vector *z = new IPM_Vector[c];
	IPM_Vector *u = new IPM_Vector[c];
	for (int i = 0; i < c; i++)
	{
		z[i] = IPM_Vector(4);
		u[i] = IPM_Vector(2);
	}
	z[0] << 0, 0, 0, 1;
	u[0] << (184 - image.size().width / 2), (image.size().height / 2 - 585);
	z[1] << 1, 0, 0, 1;
	u[1] << (340 - image.size().width / 2), (image.size().height / 2 - 510);
	z[2] << 0, 1, 0, 1;
	u[2] << (102 - image.size().width / 2), (image.size().height / 2 - 480);
	z[3] << 0, 0, 1, 1;
	u[3] << (174 - image.size().width / 2), (image.size().height / 2 - 405);
	z[4] << 1, 0, 1, 1;
	u[4] << (349 - image.size().width / 2), (image.size().height / 2 - 343);
	z[5] << 0, 1, 1, 1;
	u[5] << (84 - image.size().width / 2), (image.size().height / 2 - 315);
	z[6] << 1, 1, 1, 1;
	u[6] << (240 - image.size().width / 2), (image.size().height / 2 - 273);
	z[7] << 0.5, 0, 0.5, 1;
	u[7] << (267 - image.size().width / 2), (image.size().height / 2 - 459);

	//

	const IPM_uint n = 3 * 4;
	const IPM_uint m = c;
	const IPM_uint p = 1;

	//

	IPM_Vector x(n);
	IPM_Vector f(n);
	IPM_Matrix *a_G = new IPM_Matrix[m];
	IPM_Vector *a_h = new IPM_Vector[m];
	IPM_Vector *a_c = new IPM_Vector[m];
	IPM_Single *a_d = new IPM_Single[m];
	IPM_Matrix A(p, n);
	IPM_Vector b(p, 1);

	//

	//x.setZero();
	x.setOnes();
	//x.setRandom();

	//

	f.setZero();

	//

	const IPM_Scalar gamma = 8;

	for (IPM_uint i = 0; i < c; i++)
	{
		IPM_uint ni = 2;
		a_G[i] = IPM_Matrix(ni, n);
		a_h[i] = IPM_Vector(ni);
		a_c[i] = IPM_Vector(n);
		a_d[i] = IPM_Single();
		a_G[i].setZero();
		a_h[i].setZero();
		a_c[i].setZero();
		a_d[i].setZero();

		a_G[i].block(0, 0, 1, 4) = z[i].transpose();
		a_G[i].block(1, 4, 1, 4) = z[i].transpose();
		a_G[i].block(0, 8, 1, 4) = -u[i](0) * z[i].transpose();
		a_G[i].block(1, 8, 1, 4) = -u[i](1) * z[i].transpose();
		a_c[i].segment(8, 4) = gamma * z[i];
	}

	//

	A.setZero();
	b.setZero();

	A(0, 11) = 1;
	b(0) = 1;

	//

	class mSOCP : public SOCP {
	public:
		mSOCP()
		{
			//m_loop = 64;
			//m_bloop = 128;
			//m_slack = 8;

			//m_eps_bd = 1;
		}
#if 0
	protected:
		IPM_Error objective(const IPM_Vector& x, IPM_Single& f_o)
		{
			f_o(0, 0) = (x(8) * x(8) + x(9) * x(9) + x(10) * x(10)) / 2.0;

			return NULL;
		}

		IPM_Error Dobjective(const IPM_Vector& x, IPM_Vector& Df_o)
		{
			Df_o.setZero();
			Df_o(8) = x(8);
			Df_o(9) = x(9);
			Df_o(10) = x(10);

			return NULL;
		}

		IPM_Error DDobjective(const IPM_Vector& x, IPM_Matrix& DDf_o)
		{
			DDf_o.setZero();
			DDf_o(8, 8) = 1;
			DDf_o(9, 9) = 1;
			DDf_o(10, 10) = 1;

			return NULL;
		}
#endif
#if 1
	protected:
		IPM_Error objective(const IPM_Vector& x, IPM_Single& f_o)
		{
			const IPM_uint _n = x.size() - m_m;

			f_o.setZero();

			for (IPM_uint r = 0; r < m_m; r++)
			{
				IPM_Vector tmp = m_p_G[r] * x.head(_n);
				f_o(0, 0) += tmp.squaredNorm();
			}

			return NULL;
		}

		IPM_Error Dobjective(const IPM_Vector& x, IPM_Vector& Df_o)
		{
			const IPM_uint _n = x.size() - m_m;

			Df_o.setZero();

			for (IPM_uint r = 0; r < m_m; r++)
			{
				IPM_Vector tmp1 = m_p_G[r] * x.head(_n);
				IPM_Vector tmp2 = 2 * m_p_G[r].transpose() * tmp1;
				Df_o.head(_n) += tmp2;
			}

			return NULL;
		}

		IPM_Error DDobjective(const IPM_Vector& x, IPM_Matrix& DDf_o)
		{
			const IPM_uint _n = x.size() - m_m;

			DDf_o.setZero();

			for (IPM_uint r = 0; r < m_m; r++)
			{
				DDf_o.topLeftCorner(_n, _n) += 2 * m_p_G[r].transpose() * m_p_G[r];
			}

			return NULL;
		}
#endif
	} instance;

	ofstream ofs("logSOCP.txt");
	instance.setLog(&ofs);

	IPM_Error err = instance.solve(x, f, a_G, a_h, a_c, a_d, m, A, b);

	if (err)
	{
		cout << "!!!!! " << err << endl;
	}
	else
	{
		cout << x << endl;
		cout << "converged: " << instance.isConverged() << endl;
	}

	ofs.close();

	//

#if 1
	IPM_Matrix P(3, 4);
	P.row(0) = x.segment(0, 4).transpose();
	P.row(1) = x.segment(4, 4).transpose();
	P.row(2) = x.segment(8, 4).transpose();
	IPM_Vector v3(4);
	IPM_Vector v2;

#if 0
	v3 << 0, 0, 0, 1;
	v2 = P * v3;
	v2(0) = v2(0) / v2(2) + image.size().width / 2;
	v2(1) = image.size().height / 2 - v2(1) / v2(2);
	circle(image, Point(v2(0), v2(1)), 5, Scalar(0, 0, 255), 3);

	v3 << csz.width - 1, 0, 0, 1;
	v2 = P * v3;
	v2(0) = v2(0) / v2(2) + image.size().width / 2;
	v2(1) = image.size().height / 2 - v2(1) / v2(2);
	circle(image, Point(v2(0), v2(1)), 5, Scalar(0, 0, 255), 3);

	v3 << 0, csz.height - 1, 0, 1;
	v2 = P * v3;
	v2(0) = v2(0) / v2(2) + image.size().width / 2;
	v2(1) = image.size().height / 2 - v2(1) / v2(2);
	circle(image, Point(v2(0), v2(1)), 5, Scalar(0, 0, 255), 3);

	v3 << csz.width - 1, csz.height - 1, 0, 1;
	v2 = P * v3;
	v2(0) = v2(0) / v2(2) + image.size().width / 2;
	v2(1) = image.size().height / 2 - v2(1) / v2(2);
	circle(image, Point(v2(0), v2(1)), 5, Scalar(0, 0, 255), 3);
#endif
	for (int i = 0; i < 8; i++)
	{
		v3 << ((i & 1) ? 1 : 0), ((i & 2) ? 1 : 0), ((i & 4) ? 1 : 0), 1;
		v2 = P * v3;
		v2(0) = v2(0) / v2(2) + image.size().width / 2;
		v2(1) = image.size().height / 2 - v2(1) / v2(2);
		circle(image, Point(v2(0), v2(1)), gamma, Scalar(0, 0, 255), 1);
	}

	namedWindow("debug");
	imshow("debug", image);
	waitKey(0);
#endif

	delete[] a_G;
	delete[] a_h;
	delete[] a_c;
	delete[] a_d;
	delete[] z;
	delete[] u;
}

int main(int argc, char **argv)
{
	//testChess();
	//testHough();
	testSOCP();

	return 0;
}
