import streamlit as st


st.set_page_config(layout="wide")

r'''
## Задача Дирихле для уравнения Пуассона

Для указанного аналитического рещения $\bm u_a$ решается задача
$$
\begin{aligned}
	& - \operatorname{div} \left( k \operatorname{grad} \bm u \right) = \bm f, \quad &x& \in \Omega
	\\
	& \bm u = \bm u_a, \quad &x& \in \partial \Omega
\end{aligned}
$$

$$
k =
\begin{bmatrix}
   a & b \\
   c & d
\end{bmatrix}
$$

$\bm f$ вычисляется с помощью аналитического решения $\bm u_a$

Область
$$
\Omega = \{ x \ |\ 0 \le x_1 \le 1,\ 0 \le x_2 \le 0.75 \} = [0,\ 1] \times [0,\ 0.75]
$$
'''
