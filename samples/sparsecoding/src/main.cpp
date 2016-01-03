
#ifdef _MSC_BUILD
#include "../vs2013/opencv_2.4.9_lib.hpp"
#endif

#include <opencv2/opencv.hpp>
using namespace cv;


#include "QCQP.h"
#include "QP.h"


#include <iostream>
#include <fstream>
#include <cstdio>
using namespace std;


class Main
{
public:
	explicit Main(int blkSize, int scale);
	~Main();
	int run(void);
	void makeDict(int angle_div, int shift_div);
	void showDict(void);

	void initQCQP(IPM_Scalar weight);
	void initQP(IPM_Scalar weight);
	typedef enum {
		SOLVER_QCQP = 0,
		SOLVER_QP
	} SolverType;
	void solve(Mat &blk, SolverType type);
	void encode(SolverType type);

	void decode(string coefname);

private:
	int m_blkSize;
	vector<Mat> m_dict;

	int m_scale;
	vector<Mat> m_dictScale;

	IPM_Matrix m_D;
	IPM_Vector m_y;

	QCQP m_QCQP;
	ofstream m_ofsLog;
	IPM_Vector m_x;
	IPM_Matrix *m_a_P;
	IPM_Vector *m_a_q;
	IPM_Single *m_a_r;
	IPM_Matrix m_A;
	IPM_Vector m_b;

	QP m_QP;
	IPM_Matrix m_P;
	IPM_Vector m_q;
	IPM_Single m_r;
	IPM_Matrix m_G;
	IPM_Vector m_h;

	ofstream m_ofsCoef;

	static double thresh(double x) { return (fabs(x) < 0.5) ? 0 : x; }
};

Main::Main(int blkSize, int scale) :
	m_blkSize(blkSize), m_scale(scale)
{
	m_a_P = NULL;
	m_a_q = NULL;
	m_a_r = NULL;
	namedWindow("debug");
}

Main::~Main()
{
	if (m_a_P) delete[] m_a_P;
	if (m_a_q) delete[] m_a_q;
	if (m_a_r) delete[] m_a_r;

	m_ofsLog.close();
	m_ofsCoef.close();
}

void Main::makeDict(int angle_div, int shift_div)
{
	Mat basis(m_blkSize * 2, m_blkSize * 2, CV_32FC1);
	Mat block(m_blkSize * 2, m_blkSize * 2, CV_32FC1);
	Mat blk(m_blkSize, m_blkSize, CV_32FC1);

	Mat basisScale(m_blkSize * 2 * m_scale, m_blkSize * 2 * m_scale, CV_32FC1);
	Mat blockScale(m_blkSize * 2 * m_scale, m_blkSize * 2 * m_scale, CV_32FC1);
	Mat blkScale(m_blkSize * m_scale, m_blkSize * m_scale, CV_32FC1);

	//// DC

	rectangle(blk, Rect(0, 0, m_blkSize, m_blkSize), Scalar(1), CV_FILLED);
	m_dict.push_back(blk.clone());

	rectangle(blkScale, Rect(0, 0, m_blkSize * m_scale, m_blkSize * m_scale), Scalar(1), CV_FILLED);
	m_dictScale.push_back(blkScale.clone());

	//// Gradation
#if 0
	for (int x = 0; x < m_blkSize; x++)
	{
		line(blk, Point(x, 0), Point(x, m_blkSize - 1), Scalar(2.0 * x / (m_blkSize - 1) - 1.0));
	}
	m_dict.push_back(blk.clone());

	for (int x = 0; x < m_blkSize * m_scale; x++)
	{
		line(blkScale, Point(x, 0), Point(x, m_blkSize * m_scale - 1), Scalar(2.0 * x / (m_blkSize * m_scale - 1) - 1.0));
	}
	m_dictScale.push_back(blkScale.clone());

	for (int y = 0; y < m_blkSize; y++)
	{
		line(blk, Point(0, y), Point(m_blkSize - 1, y), Scalar(2.0 * y / (m_blkSize - 1) - 1.0));
	}
	m_dict.push_back(blk.clone());

	for (int y = 0; y < m_blkSize * m_scale; y++)
	{
		line(blkScale, Point(0, y), Point(m_blkSize * m_scale - 1, y), Scalar(2.0 * y / (m_blkSize * m_scale - 1) - 1.0));
	}
	m_dictScale.push_back(blkScale.clone());
#endif
	//// Line
#if 1
	for (int a = 0; a < angle_div; a++)
	{
		double angle = a * 360.0 / angle_div;
		Mat rot = getRotationMatrix2D(Point2f(m_blkSize - 0.5, m_blkSize - 0.5), angle, 1);
		Mat rotScale = getRotationMatrix2D(Point2f(m_blkSize * m_scale - 0.5, m_blkSize * m_scale - 0.5), angle, 1);

		for (int s = 0; s < shift_div; s++)
		{
			int shift = s * m_blkSize * 2 / (shift_div - 1) - m_blkSize;

			rectangle(basis, Rect(0, 0, m_blkSize * 2, m_blkSize * 2), Scalar(0), CV_FILLED);
			rectangle(basis, Rect(0, 0, m_blkSize * 2, m_blkSize + shift), Scalar(1), CV_FILLED);

			rectangle(basisScale, Rect(0, 0, m_blkSize * 2 * m_scale, m_blkSize * 2 * m_scale), Scalar(0), CV_FILLED);
			rectangle(basisScale, Rect(0, 0, m_blkSize * 2 * m_scale, (m_blkSize + shift) * m_scale), Scalar(1), CV_FILLED);

			warpAffine(basis, block, rot, block.size());
			blk = block(Rect(m_blkSize / 2, m_blkSize / 2, m_blkSize, m_blkSize));

			warpAffine(basisScale, blockScale, rotScale, blockScale.size());
			blkScale = blockScale(Rect(m_blkSize / 2 * m_scale, m_blkSize / 2 * m_scale, m_blkSize * m_scale, m_blkSize * m_scale));

			//cerr << sum(blk) << endl;
			if ((sum(blk)[0] <= 1) || (sum(blk)[0] >= (m_blkSize * m_blkSize - 1))) continue;

			m_dict.push_back(blk.clone());

			m_dictScale.push_back(blkScale.clone());
		}
	}
#endif
	////

#if 1
	cerr << m_dict.size() << endl;
#endif

	m_D = IPM_Matrix(m_blkSize * m_blkSize, m_dict.size());
	m_y = IPM_Vector(m_blkSize * m_blkSize);

	int b = 0;
	for (vector<Mat>::iterator it = m_dict.begin(); it != m_dict.end(); it++)
	{
		int i = 0;
		for (int y = 0; y < m_blkSize; y++)
		{
			for (int x = 0; x < m_blkSize; x++)
			{
				m_D(i++, b) = (*it).at<float>(Point(x, y));
			}
		}
		b++;
	}
}

void Main::showDict(void)
{
#if 0
	for (vector<Mat>::iterator it = m_dict.begin(); it != m_dict.end(); it++)
	{
		imshow("debug", (*it + 1) / 2); waitKey(0);
	}
#endif

#if 0
	for (vector<Mat>::iterator it = m_dictScale.begin(); it != m_dictScale.end(); it++)
	{
		imshow("debug", (*it + 1) / 2); waitKey(0);
	}
#endif

#if 1
	{
		const int blkH = (int)sqrt(m_dict.size());
		const int blkW = (int)(m_dict.size() / blkH + 0.5);
		Mat dictImage(blkH * m_blkSize, blkW * m_blkSize, CV_32FC1, Scalar(0));

		int c = 0;
		for (int y = 0; y < blkH; y++)
		{
			for (int x = 0; x < blkW; x++)
			{
				Mat blkScale = dictImage(Rect(x * m_blkSize, y * m_blkSize, m_blkSize, m_blkSize));
				if (c < m_dict.size())
				{
					blkScale += m_dict[c];
					c++;
				}
				//imshow("debug", (dictImage + 1) / 2); waitKey(1);
			}
		}
		imshow("debug", (dictImage + 1) / 2); waitKey(0);
	}
#endif
}

void Main::initQCQP(IPM_Scalar weight)
{
	m_ofsLog.open("logQCQP.txt");
#if 1
	m_QCQP.setLog(&m_ofsLog);
#endif

	/////

	const IPM_uint d = m_dict.size();

	/////

	const IPM_uint n = d * 2 + 1; // x, t, z
	const IPM_uint m = d * 2 + 2;
	const IPM_uint p = 0;

	m_x = IPM_Vector(n);
	m_a_P = new IPM_Matrix[m + 1];
	m_a_q = new IPM_Vector[m + 1];
	m_a_r = new IPM_Single[m + 1];
	m_A = IPM_Matrix(p, n);
	m_b = IPM_Vector(p, 1);

	m_x.setZero();
	for (IPM_uint i = 0; i <= m; i++)
	{
		m_a_P[i] = IPM_Matrix(n, n);
		m_a_q[i] = IPM_Vector(n);
		m_a_r[i] = IPM_Single();
		m_a_P[i].setZero();
		m_a_q[i].setZero();
		m_a_r[i].setZero();
	}
	m_A.setZero();
	m_b.setZero();

	/////

	//m_a_q[0].segment(d, d).setOnes();
	m_a_q[0].segment(d + 1, d - 1).setOnes();
	m_a_q[0](d * 2) = 1.0 / weight;
	m_a_P[1].topLeftCorner(d, d) = m_D.transpose() * m_D;
	m_a_q[1](d * 2) = -1;

	m_a_q[2](d * 2) = -1;
	m_a_r[2](0, 0) = m_blkSize * m_blkSize;

	for (IPM_uint i = 0; i < d; i++)
	{
		m_a_q[3 + i](i) = 1;
		m_a_q[3 + i](d + i) = -1;
		m_a_q[3 + d + i](i) = -1;
		m_a_q[3 + d + i](d + i) = -1;
	}
}

void Main::initQP(IPM_Scalar weight)
{
	m_ofsLog.open("logQP.txt");
#if 1
	m_QP.setLog(&m_ofsLog);
#endif

	/////

	const IPM_uint d = m_dict.size();

	/////

	const IPM_uint n = d + (d - 1); // x, t
	const IPM_uint m = (d - 1) * 2;
	const IPM_uint p = 0;

	m_x = IPM_Vector(n);
	m_P = IPM_Matrix(n, n);
	m_q = IPM_Vector(n);
	m_r = IPM_Single();
	m_G = IPM_Matrix(m, n);
	m_h = IPM_Vector(m, 1);
	m_A = IPM_Matrix(p, n);
	m_b = IPM_Vector(p, 1);

	m_x.setZero();
	m_P.setZero();
	m_q.setZero();
	m_r.setZero();
	m_G.setZero();
	m_h.setZero();
	m_A.setZero();
	m_b.setZero();

	/////

	m_P.topLeftCorner(d, d) = m_D.transpose() * m_D;
	m_q.tail(d - 1).setOnes();
	m_q.tail(d - 1) *= weight;
	for (IPM_uint i = 0; i < d - 1; i++)
	{
		m_G(i, 1 + i) = 1;
		m_G(i, d + i) = -1;
		m_G((d - 1) + i, 1 + i) = -1;
		m_G((d - 1) + i, d + i) = -1;
	}
}

void Main::solve(Mat &blk, SolverType type)
{
	{
		int i = 0;
		for (int y = 0; y < m_blkSize; y++)
		{
			for (int x = 0; x < m_blkSize; x++)
			{
				m_y(i++) = blk.at<unsigned char>(Point(x, y));
			}
		}
	}

	/////

	const IPM_uint d = m_dict.size();
	m_x.setOnes();

	if (type == SOLVER_QCQP)
	{
		const IPM_uint m = d * 2 + 1;

		m_a_q[1].head(d) = -m_D.transpose() * m_y;
		m_a_r[1](0, 0) = 0.5 * m_y.squaredNorm();

		//-----

		IPM_Error err = m_QCQP.solve(m_x, m_a_P, m_a_q, m_a_r, m, m_A, m_b);

		if (err)
		{
			cerr << "!!!!! " << err << endl;
		}
		else
		{
			cerr << "converged: " << m_QCQP.isConverged() << endl;
		}
	}
	else if (type == SOLVER_QP)
	{
		m_q.head(d) = -m_D.transpose() * m_y;
		m_r(0, 0) = 0.5 * m_y.squaredNorm();

		//-----

		IPM_Error err = m_QP.solve(m_x, m_P, m_q, m_r, m_G, m_h, m_A, m_b);

		if (err)
		{
			cerr << "!!!!! " << err << endl;
		}
		else
		{
			cerr << "converged: " << m_QP.isConverged() << endl;
		}
	}

	/////

	{
		m_y = m_D * m_x.head(d).unaryExpr(ptr_fun(thresh));
		m_ofsCoef << m_x.head(d).unaryExpr(ptr_fun(thresh)).transpose() << endl;
		int i = 0;
		for (int y = 0; y < m_blkSize; y++)
		{
			for (int x = 0; x < m_blkSize; x++)
			{
				blk.at<unsigned char>(Point(x, y)) = min<int>(max<int>(m_y(i), 0), 255);
				i++;
			}
		}
	}
}

void Main::encode(SolverType type)
{
	Mat image = imread("../../../../data/sparsecoding/miku4face.bmp", 0);
	m_ofsCoef.open("coef.txt");

	//imshow("debug", image);	waitKey(0);

	for (int y = 0; y + m_blkSize <= image.size().height; y += m_blkSize)
	{
		for (int x = 0; x + m_blkSize <= image.size().width; x += m_blkSize)
		{
			Mat blk = image(Rect(x, y, m_blkSize, m_blkSize));

			cerr << x << " " << y << endl;

			imshow("debug", image);	waitKey(1);
			//imshow("debug", blk); waitKey(1);
			solve(blk, type);
			//imshow("debug", blk); waitKey(0);
		}
	}

	imshow("debug", image);	waitKey(0);
}

void Main::decode(string coefname)
{
    ifstream ifs(coefname);

	const int blkH = 6;
	const int blkW = 6;
	const int d = m_dict.size();

	Mat image(blkW * m_blkSize, blkH * m_blkSize, CV_32FC1, Scalar(0));
	Mat imageScale(blkW * m_blkSize * m_scale, blkH * m_blkSize * m_scale, CV_32FC1, Scalar(0));

	for (int y = 0; y < blkH; y++)
	{
		for (int x = 0; x < blkW; x++)
		{
			Mat blk = image(Rect(x * m_blkSize, y * m_blkSize, m_blkSize, m_blkSize));
			Mat blkScale = imageScale(Rect(x * m_blkSize * m_scale, y * m_blkSize * m_scale, m_blkSize * m_scale, m_blkSize * m_scale));

			for (int c = 0; c < d; c++)
			{
				double coef;
				ifs >> coef;
				
				//cerr << coef << endl;

				blk += m_dict[c] * coef;
				blkScale += m_dictScale[c] * coef;
			}

			blk /= 256.0;
			max(blk, 0, blk);
			min(blk, 1, blk);
			blkScale /= 256.0;
			max(blkScale, 0, blkScale);
			min(blkScale, 1, blkScale);

			imshow("debug", image);	waitKey(1);
		}
	}

	imshow("debug", imageScale); waitKey(0);

	ifs.close();

	/////

#if 0
	Mat imageDiffScale(blkW * m_blkSize * m_scale, blkH * m_blkSize * m_scale, CV_32FC1, Scalar(0));
	Mat imageDiff = imread("../../../../data/sparsecoding/miku4face.bmp", 0);
	imageDiff.convertTo(imageDiff, CV_32FC1);
	imageDiff /= 256.0;
	imageDiff -= image;
	//imageDiff = imageDiff + 0.5;
	resize(imageDiff, imageDiffScale, imageDiffScale.size(), m_scale, m_scale, INTER_LINEAR);
	imageScale = imageScale + imageDiffScale;
	imshow("debug", imageDiffScale + 0.5); waitKey(0);
	imshow("debug", imageScale); waitKey(0);
#endif

#if 0
	Mat imageOrg = imread("../../../../data/sparsecoding/miku4face.bmp", 0);
	imageOrg.convertTo(imageOrg, CV_32FC1);
	imageOrg /= 256.0;
	Mat imageFiltScale(blkW * m_blkSize * m_scale, blkH * m_blkSize * m_scale, CV_32FC1, Scalar(0));

	resize(imageOrg, imageFiltScale, imageFiltScale.size(), m_scale, m_scale, INTER_LINEAR);
	imshow("debug", imageFiltScale); waitKey(0);

	for (int y = 0; y < imageFiltScale.size().height; y++)
	{
		for (int x = 0; x < imageFiltScale.size().width; x++)
		{
			Point p[4];
			p[0].x = x / m_scale;
			p[0].y = y / m_scale;
			p[3].x = p[0].x + 1;
			p[3].y = p[0].y + 1;
			p[1].x = p[3].x;
			p[1].y = p[0].y;
			p[2].x = p[0].x;
			p[2].y = p[3].y;

			double score[4];
			score[0] = abs(p[3].x * m_scale - x) * abs(p[3].y * m_scale - y);
			score[3] = abs(p[0].x * m_scale - x) * abs(p[0].y * m_scale - y);
			score[1] = abs(p[2].x * m_scale - x) * abs(p[2].y * m_scale - y);
			score[2] = abs(p[1].x * m_scale - x) * abs(p[1].y * m_scale - y);

			for (int s = 0; s < 4; s++)
			{
				Point pt = p[s];
				pt.x = min(pt.x * m_scale, imageScale.size().width - 1);
				pt.y = min(pt.y * m_scale, imageScale.size().height - 1);
				double d = abs(imageScale.at<float>(pt) - imageScale.at<float>(Point(x, y)));
				d *= m_scale * m_scale * 8;
				score[s] /= (d + 1);
			}

			double score_total = 0;
			for (int s = 0; s < 4; s++)
			{
				score_total += score[s];
			}

			double v = 0;
			for (int s = 0; s < 4; s++)
			{
				Point pt = p[s];
				pt.x = min(pt.x, imageOrg.size().width - 1);
				pt.y = min(pt.y, imageOrg.size().height - 1);
				v += imageOrg.at<float>(pt) * score[s] / score_total;
			}
			imageFiltScale.at<float>(Point(x, y)) = v;
			//imshow("debug", imageFiltScale); waitKey(1);
		}
	}
	imshow("debug", imageFiltScale); waitKey(0);
#endif


}

int Main::run(void)
{
	makeDict(8 * (3), 1 + 2 * (8));
	//makeDict(8 * (2), 1 + 2 * (3));
	showDict();

	//initQCQP(64.0);
	//initQCQP(16.0);
	//initQCQP(1.0);
	//encode(SOLVER_QCQP);

	//initQP(64.0);
	//initQP(16.0);
	//initQP(8.0);
	//initQP(1.0);
	//encode(SOLVER_QP);

	decode("../../../../data/sparsecoding/09_3_8_16_coef.txt");
	//decode("coef.txt");

	return 0;
}

int main(int argc, char **argv)
{
	Main m(8, 8);

	return m.run();
}
