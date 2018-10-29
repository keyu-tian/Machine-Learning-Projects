/* BP网络 by Kevin
 *
 * Supervised Learning
 * for Classification (Multi-class Classification)
 *
 * 矩阵类：自敲，模板类（便于编译器检查行列匹配等问题）
 * 隐含层层数：两层
 * 隐含层神经元个数：自定
 * Tranint Data：4位二进制数 <-> 对应位置
 * 前馈调节方式：梯度下降法
 */


#include <iostream>
#include <fstream>

#include "BP_01.h"


void read_std_data(void);
void print_std_data(void);

InputsType  stdInputs;
OutputsType stdOutputs;

void read_std_data(void)
{
	Matrix<1, INPUT_AMT> tInput;
	Matrix<1, OUTPUT_AMT> tOutput;

	std::ifstream fin("in.txt");
	if (fin.is_open())
	{
		for (int i=0; i<TEST_AMT; ++i)
		{
			for (int t=0; t<INPUT_AMT; ++t)
				fin >> tInput[0][t];
			for (int t=0; t<OUTPUT_AMT; ++t)
				fin >> tOutput[0][t];
			stdInputs[i] = tInput;
			stdOutputs[i] = tOutput;
		}
		fin.close();
	}
}

void print_std_data(void)
{
	for (int i=0; i<TEST_AMT; ++i)
	{
		for (int t=0; t<INPUT_AMT; ++t)
			std::cout << stdInputs[i][0][t] << ' ';
		for (int t=0; t<OUTPUT_AMT; ++t)
			std::cout << stdOutputs[i][0][t] << ' ';
		std::cout << '\n';
	}
}

int main()
{
	srand(time(nullptr));

	read_std_data();

	BP<8, 8> bp;
	bp.random_init_better(50000);

	bp.train();
}
