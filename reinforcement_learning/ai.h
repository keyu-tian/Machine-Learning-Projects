#ifndef AI_H
#define AI_H

#include "constant.h"
#include "matrix.h"
#include "point.h"
#include "size_params.h"


/**
 * @brief INPUT_AMT : 14
 *
 * a_r		:	r of aircraft	[0, 1]
 * a_c		:	r of aircraft	[0, 1]
 *
 * e1_r		:	r of enemy1		-1 or [0, 1]
 * e1_c		:	r of enemy1		-1 or [0, 1]
 * e2_r		:	r of enemy2		-1 or [0, 1]
 * e2_c		:	r of enemy2		-1 or [0, 1]
 *
 * b1_r		:	r of bullet1	-1 or [0, 1]
 * b1_c		:	c of bullet1	-1 or [0, 1]
 * b2_r		:	r of bullet2	-1 or [0, 1]
 * b2_c		:	c of bullet2	-1 or [0, 1]
 * b3_r		:	r of bullet3	-1 or [0, 1]
 * b3_c		:	c of bullet3	-1 or [0, 1]
 *
 * can_f	:	can fire or not	0 or 1
 * r_time	:	rest time		[0, 1]
 *
 */
constexpr int INPUT_AMT(14);


/**
 * @brief OUTPUT_AMT : 3
 *
 * left		:	move left		[?, ?]
 * right	:	move right		[?, ?]
 * fire		:	fire			[?, ?]
 *
 */
constexpr int OUTPUT_AMT(3);
enum AI_CHOICE_INDEX
{
	LEFT_INDEX = 0,
	RIGHT_INDEX = 1,
	FIRE_INDEX = 2
};


enum NEURON_AMOUNT : int
{
	TEST_AMT = 8,
	L1_AMT = 16,
	L2_AMT = 16
};

#define DEFAULT_LEARNING_RATE 0.001
#define DEFAULT_UPDATE_REWARD_RATE 0.95
#define DEFAULT_SET_REWARD_RATE 0.005

struct Observation
{
	double a_r, a_c;
	double e1_r, e1_c, e2_r, e2_c;
	double b1_r, b1_c, b2_r, b2_c, b3_r, b3_c;
	double can_f, r_time;
};


class AI
{
	private:

		double learning_rate;
		int final_move_count;
		double reward[MAX_MOVE_COUNT];

		Matrix<  1	,		L1_AMT>			B1h1;	 // 第一层的偏置系数（向量）
		Matrix<INPUT_AMT,	L1_AMT>			Wih1;	 // 输入层和第一隐含层的突触（矩阵）
		Matrix<  1	,		L1_AMT>			H1h1;	 // 第一隐含层神经元（向量）
		Matrix<  1	,		L2_AMT>			B1h2;	 // 第二层的偏置系数（向量）
		Matrix<L1_AMT,		L2_AMT>			Wh1h2;	 // 第一和第二隐含层间的突触（矩阵）
		Matrix<  1	,		L2_AMT>			H1h2;	 // 第二隐含层神经元（向量）
		Matrix<  1	,		OUTPUT_AMT>		B1o;	 // 输出层的偏置系数（向量）
		Matrix<L2_AMT,		OUTPUT_AMT>		Wh2o;	 // 第二隐含层和输出层的突触（矩阵）

		// 每个 trajectory 的偏导矩阵数据
		Matrix<  1	,		L1_AMT>			dB1h1[MAX_MOVE_COUNT];
		Matrix<INPUT_AMT,	L1_AMT>			dWih1[MAX_MOVE_COUNT];
		Matrix<  1	,		L2_AMT>			dB1h2[MAX_MOVE_COUNT];
		Matrix<L1_AMT,		L2_AMT>			dWh1h2[MAX_MOVE_COUNT];
		Matrix<  1	,		OUTPUT_AMT>		dB1o[MAX_MOVE_COUNT];
		Matrix<L2_AMT,		OUTPUT_AMT>		dWh2o[MAX_MOVE_COUNT];
		Matrix<  1	,		L1_AMT>			sum_dB1h1;
		Matrix<INPUT_AMT,	L1_AMT>			sum_dWih1;
		Matrix<  1	,		L2_AMT>			sum_dB1h2;
		Matrix<L1_AMT,		L2_AMT>			sum_dWh1h2;
		Matrix<  1	,		OUTPUT_AMT>		sum_dB1o;
		Matrix<L2_AMT,		OUTPUT_AMT>		sum_dWh2o;


	public:

		AI(void);

		void random_init(void);

		void train(int this_turn);

		int get_sum_score(void);

		void training_test(void);
		void final_test(void);
		void input_test(void);

		char aiNextMove(int this_move_count, const Observation &ob);
		void updateReward(int this_move_count, int this_move_reward);
		void gameOver(int final_move_count);

		void putssd(void);
		void putsd(void);
		void putsw(void);

};

#endif // AI_H
