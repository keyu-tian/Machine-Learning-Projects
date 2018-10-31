#ifndef MATRIX_H
#define MATRIX_H

#include <cstdio>
#include <cmath>
#include <ctime>
#include <cstring>
#include <cstdlib>
#include <map>

template <int R, int C, typename Type = double>
class Matrix
{
	public:

		Type m[R][C];

	public:

		Matrix(void) { }
		Matrix(const Type val)
		{
			for (int r=0; r<R; ++r)
				for (int c=0; c<C; ++c)
					m[r][c] = val;
		}
		Matrix(const Type inf, const Type sup)
		{
			const Type interval = (sup-inf) / RAND_MAX;
			for (int r=0; r<R; ++r)
				for (int c=0; c<C; ++c)
					m[r][c] = inf + interval * rand();
		}
		Matrix(const Matrix<R, C> &o)
		{
			memcpy(m, o.m, sizeof(m));
		}

		void random_init(const Type inf, const Type sup)
		{
			const Type interval = (sup-inf) / RAND_MAX;
			for (int r=0; r<R; ++r)
				for (int c=0; c<C; ++c)
					m[r][c] = inf + interval * rand();
		}

		void random_fix(const Type d)
		{
			const Type interval = 2*d / RAND_MAX;
			for (int r=0; r<R; ++r)
				for (int c=0; c<C; ++c)
					if (rand() < RAND_MAX/4)
						m[r][c] += interval * rand() - d;
		}

		void clear(void)
		{
			memset(m, 0, sizeof(m));
		}

		inline Matrix<R, C> & operator =(const Matrix<R, C> &o)
		{
			if (this != &o)
				memcpy(m, o.m, sizeof(m));
			return *this;
		}

		inline auto operator [](int index) -> decltype(*m)
		{
			return m[index];
		}

		Matrix<C, R> transpose(void) const
		{
			Matrix<C, R> ret;
			for (int c=0; c<C; ++c)
				for (int r=0; r<R; ++r)
					ret.m[c][r] = m[r][c];
			return ret;
		}

		Matrix<C, R> operator ~(void) const
		{
			Matrix<C, R> ret;
			for (int c=0; c<C; ++c)
				for (int r=0; r<R; ++r)
					ret.m[c][r] = m[r][c];
			return ret;
		}

		Matrix<R, C> operator +(const Matrix<R, C> &o) const
		{
			Matrix<R, C> ret;

			for (int r=0; r<R; ++r)
				for (int c=0; c<C; ++c)
					ret.m[r][c] = m[r][c] + o.m[r][c];

			return ret;
		}

		const Matrix<R, C> & operator +=(const Matrix<R, C> &o)
		{
			for (int r=0; r<R; ++r)
				for (int c=0; c<C; ++c)
					m[r][c] += o.m[r][c];

			return *this;
		}

		Matrix<R, C> operator -(const Matrix<R, C> &o) const
		{
			Matrix<R, C> ret;

			for (int r=0; r<R; ++r)
				for (int c=0; c<C; ++c)
					ret.m[r][c] = m[r][c] - o.m[r][c];

			return ret;
		}

		const Matrix<R, C> & operator -=(const Matrix<R, C> &o)
		{
			for (int r=0; r<R; ++r)
				for (int c=0; c<C; ++c)
					m[r][c] -= o.m[r][c];

			return *this;
		}

		// {R, C} * {C, C1}
		template <int C1>
		Matrix<R, C1> operator *(const Matrix<C, C1> &o) const
		{
			Matrix<R, C1> ret;

			for (int r=0; r<R; ++r)
				for (int c1=0; c1<C1; ++c1)
				{
					ret.m[r][c1] = m[r][0] * o.m[0][c1];
					for (int c=1; c<C; ++c)
						ret.m[r][c1] += m[r][c] * o.m[c][c1];
				}

			return ret;
		}

		// {R, C} ^ {R, C}
		Matrix<R, C> operator ^(const Matrix<R, C> &o) const
		{
			Matrix<R, C> ret;

			for (int r=0; r<R; ++r)
				for (int c=0; c<C; ++c)
					ret.m[r][c] = m[r][c] * o.m[r][c];

			return ret;
		}

		const Matrix<R, C> & operator ^=(const Matrix<R, C> &o)
		{
			for (int r=0; r<R; ++r)
				for (int c=0; c<C; ++c)
					m[r][c] *= o.m[r][c];

			return *this;
		}

		friend Matrix<R, C> operator *(const Type val, const Matrix<R, C> &o)
		{
			Matrix<R, C> ret;
			for (int r=0; r<R; ++r)
				for (int c=0; c<C; ++c)
					ret.m[r][c] = o.m[r][c] * val;
			return ret;
		}

		Matrix<R, C> operator *(const Type val) const
		{
			Matrix<R, C> ret;
			for (int r=0; r<R; ++r)
				for (int c=0; c<C; ++c)
					ret.m[r][c] = m[r][c] * val;
			return ret;
		}

		const Matrix<R, C> & operator *=(const Type val)
		{
			for (int r=0; r<R; ++r)
				for (int c=0; c<C; ++c)
					m[r][c] *= val;
			return *this;
		}

		// {R, C} * {R1, C1}
		template<int R1, int C1>
		Matrix <R * R1, C * C1> kronecker(const Matrix<R1, C1> &o) const
		{
			Matrix<R * R1, C * C1> ret;
			for (int r=0; r<R; ++r)
				for (int c=0; c<C; ++c)
					for (int r1=0; r1<R1; ++r1)
						for (int c1=0; c1<C1; ++c1)
							ret.m[r*R1 + r1][c*C1 + c1] = m[r][c] * o.m[r1][c1];
			return ret;
		}

		Type sq_sum(void) const
		{
			Type sum = 0.0;
			for (int r=0; r<R; ++r)
				for (int c=0; c<C; ++c)
					sum += m[r][c] * m[r][c];
			return sum;
		}

		void println(void) const
		{
			for (int r=0; r<R; ++r)
			{
				printf("[ ");
				for (int c=0; c<C; ++c)
				{
					printf("%8.4f ", m[r][c]);
				}
				printf("]\n");
			}
		}

		template <class Lambda>
		void for_each(Lambda lambda)
		{
			for (int r=0; r<R; ++r)
				for (int c=0; c<C; ++c)
					lambda(m[r][c]);
		}

		static constexpr double logsig(double x)
		{
			return 1 / (1 + exp(-x));
		}

		Matrix<1, C> getRow(int r)
		{
			Matrix<1, C> row;
			memcpy(row.m[0], m[r], sizeof(m[r]));
			return row;
		}

		void addCol(int col, const Matrix<1, R> &add)
		{
			for (int r=0; r<R; ++r)
				m[r][col] += add.m[0][r];
		}

		void setCol(int col, const Matrix<1, R> &add)
		{
			for (int r=0; r<R; ++r)
				m[r][col] = add.m[0][r];
		}


		// 浠ヤ笅鍑芥暟鍧囧彧閽堝琛屽悜閲忥紒
		Matrix<1, C> actFunc(void) const
		{
			Matrix<1, C> ret;
			for (int c=0; c<C; ++c)
				ret.m[0][c] = 1 / (1 + exp(-m[0][c]));
			return ret;
		}

		Matrix<1, C> dActFunc(void) const
		{
			Matrix<1, C> ret;
			for (int c=0; c<C; ++c)
			{
				double temp = exp(m[0][c]);
				ret.m[0][c] = temp / ((1+temp)*(1+temp));
			}
			return ret;
		}

		Matrix<C, C> toDia(void) const
		{
			Matrix<C, C> ret;
			memset(ret.m, 0, sizeof(ret.m));
			for (int c=0; c<C; ++c)
			{
				ret.m[c][c] = m[0][c];
			}
			return ret;
		}

		std::pair<int, double> roulette(void) const
		{
			std::pair<int, double> final_choice;

			Type probity[C], exp_sum = 0.0;
			for (int c=0; c<C; ++c)
			{
				probity[c] = exp(m[0][c]);
				exp_sum += probity[c];
			}

			for (int c=0; c<C; ++c)
				probity[c] /= exp_sum;

			double sum_probity = (double)rand() / RAND_MAX;
			for (int index=0; index<C; ++index)
			{
				sum_probity -= probity[index];
				if (sum_probity <= 0)
				{
					final_choice.first = index;
					final_choice.second = 1 - probity[index];	// 鍋忓鏁?
					break;
				}
			}

			return final_choice;
		}
};


#endif // MATRIX_H
