#include "ai.h"
#include "main_game.h"

#include <conio.h>

AI::AI() :
	learning_rate(DEFAULT_LEARNING_RATE),
	final_move_count(0)
{

}

void AI::random_init(void)
{
	H1h1.random_init(-0.5, 0.5);
	H1h2.random_init(-0.5, 0.5);
	Wih1.random_init(-0.5, 0.5);
	Wh1h2.random_init(-0.5, 0.5);
	Wh2o.random_init(-0.5, 0.5);
	B1h1.random_init(-0.25, 0.25);
	B1h2.random_init(-0.25, 0.25);
	B1o.random_init(-0.25, 0.25);
}

void AI::train(int this_turn)
{
#define TURNS1 511
#define TURNS2 4095

	if ((this_turn & TURNS1) == 0)
	{
		system("cls");
		gotoXY(0, 0);
		printf("start training BP-NeuralNetwork %d\n", this_turn);
	}

	int max_score = -1000;
	int min_score = 1000;
	int avr_score = 0;

	sum_dB1h1.clear();
	sum_dWih1.clear();
	sum_dB1h2.clear();
	sum_dWh1h2.clear();
	sum_dB1o.clear();
	sum_dWh2o.clear();

	for (int t=0; t<TEST_AMT; ++t)
	{
		memset(dB1o, 0, sizeof(dB1o));
		memset(dWh2o, 0, sizeof(dWh2o));
		memset(reward, 0, sizeof(reward));


		// one trajectory
		int this_score = AIGame(*this, this_turn, false);


		if (this_score > max_score)
			max_score = this_score;
		if (this_score < min_score)
			min_score = this_score;
		avr_score += this_score;

		// inner-summation (after every trajectory)
		// final_move_count : [0, MAX_MOVE_COUNT-1]
		for (int t=final_move_count; t>=0; --t)
		{
			if (t != final_move_count)
				reward[t] += reward[t+1] * DEFAULT_UPDATE_REWARD_RATE;

			sum_dWih1  +=  reward[t] * dWih1[t];
			sum_dWh1h2 +=  reward[t] * dWh1h2[t];
			sum_dWh2o  +=  reward[t] * dWh2o[t];
			sum_dB1h1  +=  reward[t] * dB1h1[t];
			sum_dB1h2  +=  reward[t] * dB1h2[t];
			sum_dB1o   +=  reward[t] * dB1o[t];
		}
	}

	// outer-summation (after all trajectories)
	Wih1  +=  sum_dWih1  * (learning_rate / TEST_AMT);
	Wh1h2 +=  sum_dWh1h2 * (learning_rate / TEST_AMT);
	Wh2o  +=  sum_dWh2o  * (learning_rate / TEST_AMT);
	B1h1  +=  sum_dB1h1  * (learning_rate / TEST_AMT);
	B1h2  +=  sum_dB1h2  * (learning_rate / TEST_AMT);
	B1o   +=  sum_dB1o   * (learning_rate / TEST_AMT);

	if ((this_turn & TURNS1) == 0)
	{
		printf("min: %-4d  max: %-4d  avr: %-4d\n\n", min_score, max_score, avr_score/TEST_AMT);
		puts("\nplease wait for the next iteration...\n");
	}
	if ((this_turn & TURNS2) == 0)
	{
		AIGame(*this, this_turn / TURNS2, true);

		for (int t=final_move_count; t>=0; --t)
		{
			if (t != final_move_count)
				reward[t] += reward[t+1] * DEFAULT_UPDATE_REWARD_RATE;
		}
		printReward();

		puts("\nplease wait...");
	}
}

int AI::get_sum_score(void)
{
	int sum_score = 0;

	int times = 3;
	while (times--)
		sum_score += AIGame(*this, 0, false);

	return sum_score;
}

// 鏇剧粡灏濊瘯杩囩敤杩欎釜鍑芥暟鏉ヨ繘琛宔volution鐨勭瓫閫夛紝褰撲笖浠呭綋涓ら」鍙傛暟閮芥瘮鍘熸潵鐨勫皬鐨勬椂鍊欐墠杩涘寲锛屼絾鏄彂鐜板埌鍚庢湡涔嬪悗涓嶅鏄撴弧瓒虫潯浠讹紝鏀舵暃閫熷害涓嶅拫鍦帮紝浜庢槸鑰冭檻鏀惧鏉?
//		std::pair<double, double> evaluate(void)
//		{
//			double error_sum = 0.0;
//			double max_error = 0.0;
//			for (int t=0; t<TEST_AMT; ++t)
//			{
////				const Matrix<1, OUTPUT_AMT> &O1o = (((((inputs[i] * Wih1 + B1h1).actFunc()) * Wh1h2 + B1h2).actFunc()) * Wh2o + B1o).actFunc();
//				const Matrix<1, INPUT_AMT> X1i = inputs[t];
//				const Matrix<1, OUTPUT_AMT> Y1o = outputs[t];
//
//				const Matrix<1, INPUT_AMT> O1i = X1i;
//
//				Matrix<1, L1_AMT> X1h1 = O1i * Wih1;
//				X1h1 = X1h1 + B1h1;
//				const Matrix<1, L1_AMT> O1h1 = X1h1.actFunc();
//
//				Matrix<1, L2_AMT> X1h2 = O1h1 * Wh1h2;
//				X1h2 = X1h2 + B1h2;
//				const Matrix<1, L2_AMT> O1h2 = X1h2.actFunc();
//
//				Matrix<1, OUTPUT_AMT> X1o = O1h2 * Wh2o;
//				X1o = X1o + B1o;
//				const Matrix<1, OUTPUT_AMT> O1o = X1o.actFunc();
//
//				const Matrix<1, OUTPUT_AMT> D1o = Y1o - O1o;
//
//				double cur_error = D1o.sq_sum();
//				error_sum += cur_error;
//				if (max_error < cur_error)
//					max_error = cur_error;
//			}
//			return std::pair<double, double>(max_error, error_sum);
//		}

void AI::training_test(void)
{
	for (int t=0; t<TEST_AMT; ++t)
	{
	}
	printf("\n");
}

void AI::final_test(void)
{
	for (int t=0; t<TEST_AMT; ++t)
	{
	}
	printf("\n");
}

void AI::input_test(void)
{
	Matrix<1, INPUT_AMT> in;
	printf("\nplease input %d num:\n", INPUT_AMT);
	while (~scanf("%lf", in.m[0]))
	{
	}
}


// this_move_count : 0 -> MAX_MOVE_COUNT
char AI::aiNextMove(int this_move_count, const Observation &ob)
{
	Matrix<1, INPUT_AMT> O1i;

//	O1i.m[0][0] = ob.a_r;
	O1i.m[0][0] = ob.a_c;
	O1i.m[0][1] = ob.e1_r;
	O1i.m[0][2] = ob.e1_c;
	O1i.m[0][3] = ob.e2_r;
	O1i.m[0][4] = ob.e2_c;
	O1i.m[0][5] = ob.b1_r;
	O1i.m[0][6] = ob.b1_c;
	O1i.m[0][7] = ob.b2_r;
	O1i.m[0][8] = ob.b2_c;
	O1i.m[0][9] = ob.b3_r;
	O1i.m[0][10] = ob.b3_c;
	O1i.m[0][11] = ob.can_f;
	O1i.m[0][12] = ob.r_time;

	// 鍓嶅悜浼犳挱
	const Matrix<1, L1_AMT> &I1h1 = O1i * Wih1 + B1h1;
	const Matrix<1, L1_AMT> &O1h1 = I1h1.actFunc();

	const Matrix<1, L2_AMT> &I1h2 = O1h1 * Wh1h2 + B1h2;
	const Matrix<1, L2_AMT> &O1h2 = I1h2.actFunc();

	const Matrix<1, OUTPUT_AMT> &I1o = O1h2 * Wh2o + B1o;
	const Matrix<1, OUTPUT_AMT> &O1o = I1o;

//  鍙嶉浼犳挱
//	const Matrix<1, OUTPUT_AMT> &D1o = Y1o - O1o;
//	const Matrix<1, OUTPUT_AMT> &E1o = D1o ^ (X1o.dActFunc());
	std::pair<int, double> final_choice = O1o.roulette();

	const Matrix<1, L2_AMT> &D1h2 = final_choice.second * (~Wh2o).getRow(final_choice.first);
	const Matrix<1, L2_AMT> &E1h2 = D1h2 ^ (I1h2.dActFunc());

	const Matrix<1, L1_AMT> &D1h1 = E1h2 * ~Wh1h2;
	const Matrix<1, L1_AMT> &E1h1 = D1h1 ^ (I1h1.dActFunc());

//  Policy Gradient
	dWih1[this_move_count] =  ~O1i * E1h1;
	dB1h1[this_move_count] =  E1h1;

	dWh1h2[this_move_count] =  ~O1h1 * E1h2;
	dB1h2[this_move_count] =  E1h2;

//	Wh2o += ~O1h2 * E1o;
	dWh2o[this_move_count].setCol(final_choice.first, final_choice.second * O1h2);
//	B1o += E1o;
	dB1o[this_move_count].m[0][final_choice.first] = final_choice.second;

	char next_move;
	switch (final_choice.first)
	{
		case LEFT_INDEX:
			next_move = LEFT;
			break;
		case RIGHT_INDEX:
			next_move = RIGHT;
			break;
		case FIRE_INDEX:
			next_move = FIRE;
			break;
	}

	return next_move;
}

// this_move_count : 0 -> MAX_MOVE_COUNT
void AI::updateReward(int this_move_count, int this_move_reward)
{
	reward[this_move_count] = DEFAULT_SET_REWARD_RATE * this_move_reward;
}

// final_move_count : in [0, MAX_MOVE_COUNT - 1]
void AI::gameOver(int final_move_count)
{
	this->final_move_count = final_move_count;
}

void AI::putssd()
{
	puts("s_dWih1");sum_dWih1.println();
	puts("s_dWh1h2");sum_dWh1h2.println();
	puts("s_dWh2o");sum_dWh2o.println();
	puts("s_dB1h1");sum_dB1h1.println();
	puts("s_dB1h2");sum_dB1h2.println();
	puts("s_dB1o");sum_dB1o.println();
}

void AI::putsw()
{
	printf("\nWih1 [%d x %d]:\n", INPUT_AMT, L1_AMT);Wih1.println();
	printf("\nWh1h2 [%d x %d]:\n", L1_AMT, L2_AMT);Wh1h2.println();
	printf("\nWh2o [%d x %d]:\n", L2_AMT, OUTPUT_AMT);Wh2o.println();
	puts("");
	printf("\nB1h1 [%d x %d]:\n", 1, L1_AMT);B1h1.println();
	printf("\nB1h2 [%d x %d]:\n", 1, L2_AMT);B1h2.println();
	printf("\nB1o [%d x %d]:\n", 1, OUTPUT_AMT);B1o.println();
	puts("");
}

void AI::printReward()
{
	printf("history rewards table of this trajectory: (sum : %d)\n", final_move_count + 1);
	for (int i=0; i<=final_move_count; ++i)
	{
		if ((i & 15) == 0)
			printf("%3d - %3d : ", i/16 * 16, i/16 * 16 + 15);
		if (reward[i])
			printf("%6.2f  ", reward[i] / DEFAULT_SET_REWARD_RATE);
		else
			printf("   --   ");
		if (((i+1) & 15) == 0)
			putchar('\n');
	}
	puts("");
}
