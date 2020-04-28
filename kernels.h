#ifndef KERNELS_H
#define KERNELS_H
#include <array>

template<class T, size_t Rows, size_t Cols> using matrix = std::array<std::array<T, Cols>, Rows>;

float sobelx[3][3] =  { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
float sobely[3][3] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };
float prewittx[3][3] = { {-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1} };
float prewitty[3][3] = { {-1, -1, -1}, {0, 0, 0}, {1, 1, 1} };
float robertsx[2][2] = { {1, 0}, {0, -1} };
float robertsy[2][2] = { {0, 1}, {-1, 0} };
float scharrx[3][3] = { {-3, 0, 3}, {-10, 0, 10}, {-3, 0, 3} };
float scharry[3][3] = { {-3, -10, -3}, {0, 0, 0}, {3, 10, 3} };
float sobel5x[5][5] = { {-1, -2, 0, 2, 1}, {-4, -8, 0, 8, 4}, {-6, -12, 0, 12, 6}, {-4, -8, 0, 8, 4}, {-1, -2, 0, 2, 1} };
float sobel5y[5][5] = { {1, 4, 6, 4, 1}, {2, 8, 12, 8, 2}, {0, 0, 0, 0, 0}, {-2, -8, -12, -8, -2}, {-1, -4, -6, -4, -1} };
#endif // KERNELS_H
