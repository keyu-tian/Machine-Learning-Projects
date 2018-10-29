#ifndef BP_01_H
#define BP_01_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>

#include "matrix.h"

constexpr int TEST_AMT(16);		// 数据组数
constexpr int INPUT_AMT(4);		// 输入数目
constexpr int OUTPUT_AMT(16);		// 输出数目

constexpr float LEARNING_RATE(0.05);
constexpr double ERROR_LIMIT(0.0000001);

typedef Matrix<1, INPUT_AMT>  InputsType[TEST_AMT];
typedef Matrix<1, OUTPUT_AMT> OutputsType[TEST_AMT];

extern InputsType  stdInputs;
extern OutputsType stdOutputs;

template <uint L1_AMT, uint L2_AMT>
class BP
{
	private:

		int _i = INPUT_AMT;
		int _h1 = L1_AMT;
		int _h2 = L2_AMT;
		int _o = OUTPUT_AMT;

		InputsType		&inputs;		// 训练用输入（向量）
		OutputsType		&outputs;		// 训练用输入（向量）

		Matrix<  1	,	L1_AMT>			B1h1;	 // 第一层的偏置系数（向量）
		Matrix<INPUT_AMT,	L1_AMT>			Wih1;	 // 输入层和第一隐含层的突触（矩阵）
		Matrix<  1	,	L1_AMT>			H1h1;	 // 第一隐含层神经元（向量）
		Matrix<  1	,	L2_AMT>			B1h2;	 // 第二层的偏置系数（向量）
		Matrix<L1_AMT,		L2_AMT>			Wh1h2;	 // 第一和第二隐含层间的突触（矩阵）
		Matrix<  1	,	L2_AMT>			H1h2;	 // 第二隐含层神经元（向量）
		Matrix<  1	,	OUTPUT_AMT>		B1o;	 // 输出层的偏置系数（向量）
		Matrix<L2_AMT,		OUTPUT_AMT>		Wh2o;	 // 第二隐含层和输出层的突触（矩阵）

	public:

		BP(void) :
			inputs(stdInputs), outputs(stdOutputs) { }

		void init_inputs_and_outputs(const InputsType &inputs, const OutputsType &outputs)
		{
			this->inputs  = inputs;
			this->outputs = outputs;
		}

		void random_init(void)
		{
			H1h1.random_init(-1, 1);
			H1h2.random_init(-1, 1);
			Wih1.random_init(-1, 1);
			Wh1h2.random_init(-1, 1);
			Wh2o.random_init(-1, 1);
			B1h1.random_init(-1, 1);
			B1h2.random_init(-1, 1);
			B1o.random_init(-1, 1);
		}

		void random_init_better(uint count)
		{
			printf("random selecting (%d times)...\n", count);

			auto *bp = new BP<L1_AMT, L2_AMT>[count];
			for (int i=0; i<count; ++i)
			{
				bp[i].random_init();
			}

			uint minErrorIndex = 0;
			double minErrorValue = bp[0].get_error_sum();
			double maxErrorValue = minErrorValue;

			for (int i=1; i<count; ++i)
			{
				double nowErrorValue = bp[i].get_error_sum();
				if (minErrorValue > nowErrorValue)
				{
					minErrorValue = nowErrorValue;
					minErrorIndex = i;
				}
				if (maxErrorValue < nowErrorValue)
					maxErrorValue = nowErrorValue;
			}

			const BP<L1_AMT, L2_AMT> &best_bp = bp[minErrorIndex];
			this->B1h1  =  best_bp.B1h1;
			this->Wih1  =  best_bp.Wih1;
			this->H1h1  =  best_bp.H1h1;
			this->B1h2  =  best_bp.B1h2;
			this->Wh1h2 =  best_bp.Wh1h2;
			this->H1h2  =  best_bp.H1h2;
			this->B1o   =  best_bp.B1o;
			this->Wh2o  =  best_bp.Wh2o;

			printf("best: %f, worst: %f\n", minErrorValue, maxErrorValue);
			puts("random select OK!");
			delete []bp;
		}

		void evolution(uint count)
		{
			int minErrorIndex = -1;
			double minErrorValue = get_error_sum();

			BP<L1_AMT, L2_AMT> bp[count];

			for (int i=0; i<count; ++i)
			{
				bp[i].B1h1  =  this->B1h1;
				bp[i].B1h1.random_fix(0.01);
				bp[i].Wih1  =  this->Wih1;
				bp[i].Wih1.random_fix(0.01);
				bp[i].H1h1  =  this->H1h1;
				bp[i].H1h1.random_fix(0.01);
				bp[i].B1h2  =  this->B1h2;
				bp[i].B1h2.random_fix(0.01);
				bp[i].Wh1h2 =  this->Wh1h2;
				bp[i].Wh1h2.random_fix(0.01);
				bp[i].H1h2  =  this->H1h2;
				bp[i].H1h2.random_fix(0.01);
				bp[i].B1o   =  this->B1o;
				bp[i].B1o.random_fix(0.01);
				bp[i].Wh2o  =  this->Wh2o;
				bp[i].Wh2o.random_fix(0.01);

				double nowErrorValue = bp[i].get_error_sum();
				if (minErrorValue > nowErrorValue)
				{
					minErrorValue = nowErrorValue;
					minErrorIndex = i;
				}
			}

			if (minErrorIndex != -1)	// 变异出了更好的
			{
				this->B1h1  =  bp[minErrorIndex].B1h1;
				this->Wih1  =  bp[minErrorIndex].Wih1;
				this->H1h1  =  bp[minErrorIndex].H1h1;
				this->B1h2  =  bp[minErrorIndex].B1h2;
				this->Wh1h2 =  bp[minErrorIndex].Wh1h2;
				this->H1h2  =  bp[minErrorIndex].H1h2;
				this->B1o   =  bp[minErrorIndex].B1o;
				this->Wh2o  =  bp[minErrorIndex].Wh2o;
			}
		}

		void train(double rate = LEARNING_RATE, double error_limit = ERROR_LIMIT)
		{
			puts("\nstart training:\n");
			int times = 0;
			while (true)
			{
				evolution(32);
				++times;
				double error = 0;
				for (int t=0; t<TEST_AMT; t++)
				{
					const Matrix<1, INPUT_AMT> X1i = inputs[t];
					const Matrix<1, OUTPUT_AMT> Y1o = outputs[t];

					const Matrix<1, INPUT_AMT> O1i = X1i;

					Matrix<1, L1_AMT> X1h1 = O1i * Wih1;
					X1h1 = X1h1 + B1h1;
					const Matrix<1, L1_AMT> O1h1 = X1h1.actFunc();

					Matrix<1, L2_AMT> X1h2 = O1h1 * Wh1h2;
					X1h2 = X1h2 + B1h2;
					const Matrix<1, L2_AMT> O1h2 = X1h2.actFunc();

					Matrix<1, OUTPUT_AMT> X1o = O1h2 * Wh2o;
					X1o = X1o + B1o;
					const Matrix<1, OUTPUT_AMT> O1o = X1o.actFunc();

					const Matrix<1, OUTPUT_AMT> D1o = Y1o - O1o;
					Matrix<1, OUTPUT_AMT> E1o = D1o;
					E1o ^= X1o.dActFunc();

					const Matrix<OUTPUT_AMT, L2_AMT> Woh2 = ~Wh2o;
					const Matrix<1, L2_AMT> D1h2 = E1o * Woh2;
					Matrix<1, L2_AMT> E1h2 = D1h2;
					E1h2 ^= X1h2.dActFunc();

					const Matrix<L2_AMT, L1_AMT> Wh2h1 = ~Wh1h2;
					const Matrix<1, L1_AMT> D1h1 = E1h2 * Wh2h1;
					Matrix<1, L1_AMT> E1h1 = D1h1;
					E1h1 ^= X1h1.dActFunc();

					error += D1o.sq_sum() / 2;

					Wih1 += rate * ((~O1i).kronecker(E1h1));
					B1h1 += rate * E1h1;

					Wh1h2 += rate * ((~O1h1).kronecker(E1h2));
					B1h2 += rate * E1h2;

					Wh2o += rate * ((~O1h2).kronecker(E1o));
					B1o  += rate * E1o;
				}
				if ((times & 4095) == 0 || times == 1)
				{
					printf("迭代次数：%d，均方误差：%f，各自输出: \n", times, error);
					training_test();
				}
				if (error < error_limit)
				{
					printf("\n====================== 训练完成！======================\n迭代次数：%d，均方误差：%g\n绝对误差: \n", times, error);
					final_test();
					break;
				}
			}
		}

		double get_error_sum(void)
		{
			double error_sum = 0.0;
			for (int t=0; t<TEST_AMT; ++t)
			{
				const Matrix<1, INPUT_AMT> X1i = inputs[t];
				const Matrix<1, OUTPUT_AMT> Y1o = outputs[t];

				const Matrix<1, INPUT_AMT> O1i = X1i;

				Matrix<1, L1_AMT> X1h1 = O1i * Wih1;
				X1h1 = X1h1 + B1h1;
				const Matrix<1, L1_AMT> O1h1 = X1h1.actFunc();

				Matrix<1, L2_AMT> X1h2 = O1h1 * Wh1h2;
				X1h2 = X1h2 + B1h2;
				const Matrix<1, L2_AMT> O1h2 = X1h2.actFunc();

				Matrix<1, OUTPUT_AMT> X1o = O1h2 * Wh2o;
				X1o = X1o + B1o;
				const Matrix<1, OUTPUT_AMT> O1o = X1o.actFunc();

				const Matrix<1, OUTPUT_AMT> D1o = Y1o - O1o;

				error_sum += D1o.sq_sum();
			}
			return error_sum;
		}

		void training_test(void)
		{
			for (int t=0; t<TEST_AMT; ++t)
			{
				const Matrix<1, INPUT_AMT> X1i = inputs[t];
				const Matrix<1, OUTPUT_AMT> Y1o = outputs[t];

				const Matrix<1, INPUT_AMT> O1i = X1i;

				Matrix<1, L1_AMT> X1h1 = O1i * Wih1;
				X1h1 = X1h1 + B1h1;
				const Matrix<1, L1_AMT> O1h1 = X1h1.actFunc();

				Matrix<1, L2_AMT> X1h2 = O1h1 * Wh1h2;
				X1h2 = X1h2 + B1h2;
				const Matrix<1, L2_AMT> O1h2 = X1h2.actFunc();

				Matrix<1, OUTPUT_AMT> X1o = O1h2 * Wh2o;
				X1o = X1o + B1o;
				const Matrix<1, OUTPUT_AMT> O1o = X1o.actFunc();

				printf("[ ");
				for (int t=0; t<OUTPUT_AMT; ++t)
					if(fabs(O1o.m[0][t]) > 0.0001)printf("%.4f ", O1o.m[0][t]);
					else printf(" ----  ");
				printf("]\n");
			}
			printf("\n");
		}

		void final_test(void)
		{
			for (int t=0; t<TEST_AMT; ++t)
			{
				const Matrix<1, INPUT_AMT> X1i = inputs[t];
				const Matrix<1, OUTPUT_AMT> Y1o = outputs[t];

				const Matrix<1, INPUT_AMT> O1i = X1i;

				Matrix<1, L1_AMT> X1h1 = O1i * Wih1;
				X1h1 = X1h1 + B1h1;
				const Matrix<1, L1_AMT> O1h1 = X1h1.actFunc();

				Matrix<1, L2_AMT> X1h2 = O1h1 * Wh1h2;
				X1h2 = X1h2 + B1h2;
				const Matrix<1, L2_AMT> O1h2 = X1h2.actFunc();

				Matrix<1, OUTPUT_AMT> X1o = O1h2 * Wh2o;
				X1o = X1o + B1o;
				const Matrix<1, OUTPUT_AMT> O1o = X1o.actFunc();

				printf("[ ");
				for (int t=0; t<OUTPUT_AMT; ++t)
					printf("%.4f ", O1o.m[0][t]);
				printf("]\n");
			}
			printf("\n");
		}
};


#endif	// BP_01_H
