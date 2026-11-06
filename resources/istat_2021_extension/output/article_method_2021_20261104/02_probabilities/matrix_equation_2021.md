# New2021probabilitymatrix

Let a be building age in years at2021 and s dwelling area in m2. The ordered basis is exactly the CSV column order:

$$x=[1, a/100, s/100, as/10000, a^2/10000, s^2/10000]^T.$$

$$\hat p = Bx, \quad B=\begin{bmatrix}
0.0649592 & 0.5497158 & 0.0000000 & 0.0000000 & -0.2697043 & 0.0000000 \\ 
0.3951931 & -0.5505812 & 0.1030504 & 0.0000000 & 0.3084934 & 0.0000000 \\ 
0.4571235 & -0.0889840 & -0.2134687 & 0.0000000 & 0.0000000 & 0.0000000 \\ 
0.1527772 & 0.0000000 & 0.0000000 & 0.0000000 & 0.0000000 & 0.0791919
\end{bmatrix}.$$

Rows are aligned archetypesA1,A2,A3,A4. Zeros represent terms removed by backwards elimination. This is a new2021fit, not the published2013matrix. Primary fit uses equal-household empirical window ratios, support>=5,age<150 andarea<150. The expression is unconstrained OLS; use saved range/sum flags before interpreting it as probabilities, particularly outside the primary domain. No automatic clipping or renormalization is applied.
