# Detecting Financial Fraud with First Digit Law
#### Feng Xi, 2201212354
## 1. Introduction to First Digit Law
> Have you found that, numbers in real life usually begin with 1, and seldom begin with 9?  

> First Digit Law, or Benford's Law, tells us that, if we collect the first digit of positive numbers, we will probably see the following distribution:

<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/537fc08a-48d1-47ed-80db-c8f2d463277d/to/image" width="600" height="400" alt="Benford's Law with stock price">
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/8ea6ed32-53a4-4d11-9bcc-05b79ff35ac7/to/image" width="500" height="300" alt="Benford's Law with GDP">
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/8612709a-5c43-40fb-ac83-d2288009b643/to/image" width="400" height="300" alt="Benford's Law with physics constants">
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/c3dbb42f-5594-4e22-ab67-75cd94a5058d/to/image" width="500" height="300" alt="Benford's Law with many numbers">

First digit Law: 
> The probability that a positive number in b-base begins with a digit in [d,d+l) can be expressed as:
>
> $$ P_{b,l,d}= \log_b⁡(1+\frac{l}{d}) $$
>
> We let $b=10$ (in decimal system) and $l=1$,
> 
> $$ P_{d}= \log_{10}⁡(1+\frac{1}{d}) $$
> 
> ![图片](https://github.com/pkujackfeng/MLF_final_project/assets/90912432/b24c1d01-d861-4a41-a7e3-a2499bc49421)

## 2. Proof of First Digit Law
### 2.1 Intuition
> To grow from 1 to 2, you need to grow by 100%.
> 
> To grow from 8 to 9, you just need to grow by 12.5%.
>
> So it's more difficult to grow from 1 to 2, which means there are more time that you stay in [1,2).
>
### 2.2 Proof with Laplace Transform
> Reference: [First Digit Law from Laplace Transform](https://www.sciencedirect.com/science/article/pii/S0375960119302452)

> F(x) : PDF (probability density function)
> 
> $P_d$: the probability that $x \in \[d\cdot 10^n,(d+1)\cdot 10^n\)$
> 
> $$ P_{d}= \sum_{n=-\infty}^{+\infty}\int_{d\cdot 10^n}^{(d+1)\cdot 10^n} F(x) {\rm{d}} x $$
>
> $$ =\int_0^{\infty}F(x)g_d(x){\rm{d}}x$$
> 
> where
>
> $$ g_d(x) = \sum_{n=-\infty}^{+\infty} \[\eta(x-d\cdot 10^n)-\eta(x-(d+1)\cdot 10^n)\] $$
>
> $$ \eta(x) = \begin{cases}
1 \quad {\rm{if}} \quad  x\geq 0 \\
0 \quad {\rm{if}} \quad  x>0 \\
\end{cases}
$$

> In interval [1,30),  the gap between the shaded areas in $g_2(x)$ is wider than that in $g_2(x)$
> 
> ![图片](https://github.com/pkujackfeng/MLF_final_project/assets/90912432/07756d3f-d84f-4c51-bd61-7d44267ab06c)
>
> Above intuitively explains the inequality among the 9 digits, where smaller leading digits are more likely to appear.

> The idea of the proof is to use
>
> - G(x): the Laplace transform of g(x)
>
> - f(x): the inverse Laplace transform of F(x),
>
> - and the property of Laplace Transform:
> 
> $$ \int_{0}^{+\infty} F(x)g(x) {\rm{d}} x = \int_{0}^{+\infty} f(t)G(t) {\rm{d}} t $$
>
>  and finally proved that:
>
> $$ P_d = \log\(1+\frac{1}{d}\) + \int_{-\infty}^{+\infty}\tilde{f} (s) \tilde{\Delta} (s) {\rm{d}}s$$
>
> where the second term is a small error term.

### 2.3 Proof with Fourier Transform
> My undergraduate thesis is to prove First Digit Law with Fourier Transform.
> 
> Basic idea is similar.
> 
> The proof with Laplace/Fourier transform requires that the PDF of x should meet the requirements of having Laplace/Fourier Transform.
> 
>  The requiments of having Fourier Transform are lower than that of having Laplace Transform.
>
## 3. Motivation
> First Digit Law can be used to detect financial Fraud, because numbers in financial statements also follow the First Digit Law. If not, there is possibility that someone manupulates the numbers.

> Below we show the first digit distribution of all the positive numbers in the 2021Q3 quaterly financial statements of two companies BYD and Gotion High-TECH, who both make EV batteries:
>
> ![图片](https://github.com/pkujackfeng/MLF_final_project/assets/90912432/898d67ac-352e-4ec9-a248-a9acf5907169)
>
> Gotion's distribution seems to violate the First Digit Law, and was really caught financial fraud in July 2022.

## 4. Data
### 4.1 Variable
> X: the difference between
>
> - the distribution of the first digits in a company's 3 financial statements
>
> - and the Benford distribution

> y: whether the company was reported financial fraud
>
> - 1: Yes
>
> - 0: No
>
> y depends on the auditor's opinion on the financial statements:
> 
>   - y = 0, Standard unqualified opinion
>
>   - y = 1, Unqualified opinion with emphasis paragraph
>   
>   - y = 1, reserved opinion
>   
>   - y = 1, inability to express opinion
>   
>  - y = 1, negative opinion
>
### 4.2 Time period
> X: annual financial statements in 2015~2019

> y: whether the company was reported financial fraud in 2015~2022
>
> - The average time interval between a company's financial fraud and its discovery is 2.97 years
>
> Source: *Research on Financial Fraud Identification of Listed Companies Based

### 4.3 Companies
> We choose CSI 500 excluding finance stocks:
>
> - CSI 500: China Securities ranking 301~800 in market cap
> - why exclude finance stocks: finance companies have many unique financial accounts, which don't apply to non-finance companies

### 4.4 Data aquisition
> We get numbers of financial statements through WIND API

> Then we compute the first digit frequency, and the distribution is as follows:
> 
> <img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/a0fffa48-7e74-49de-a35a-a62f9bcc6e88/to/image" width="600" height="400" alt="Samples_Benford's Law">

> We get auditor's opinion on financial statements through WIND EXCEL Plugger

### 4.5 Data processing
> $X_i$ describes the difference between real frequency and Benford frequency:
>
> $$ X_i = \frac{{\rm{frequency \ \ of \ \ beginning \ \ with \ \ digit \ \ i}}}{{\rm{Benford \ \  frequency\ \ [i]}}} $$

> A big issue is that the dataset is imbalanced, there are too few samples with y= 0.
>
> So within the trainning set we use *RandomUnderSampler:
>
> - Before RandomUnderSampler, 36/279=12% samples are y=1.
> - After RandomUnderSampler, 36/72=50% samples are y=1.

### 4.6 The Processed Data
> ![图片](https://github.com/pkujackfeng/MLF_final_project/assets/90912432/0d4d7635-5d1d-4448-a347-d9d282dec28e)


## 5. Model Results
### 5.1 Logistic Regression；AUC = 60%
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/50cabcce-91ae-45bc-bae4-9b229a413e61" width="400" height="250">
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/cb503e78-edb9-4193-a8f4-215be157235b" width="300" height="250">

### 5.2 MLPClassifier: AUC = 56%
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/a6477aa6-6c68-4e1a-b74f-3a9cd73fc4b4" width="400" height="250">
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/987d9a63-9332-446d-ade3-9e1caf2e9b0d" width="300" height="250">

### 5.3 SVM: AUC = 60%
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/9993f92a-d930-4105-9d6f-d7d4761bdfba" width="400" height="250">
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/0fce3638-7d3e-47ad-9f5b-523c0e1a5206" width="300" height="250">

### 5.4 Decision Tree: AUC = 56%
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/8cabaf9f-55ab-4ccd-b429-2414add0c6b4" width="400" height="250">
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/2b811aa6-5841-4813-a238-21ad56f83e23" width="300" height="250">

### 5.5 Random Forest: AUC = 61%
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/0f8daa82-f628-47c6-a34f-acdd34efe63c" width="400" height="250">
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/02114c21-0d81-4903-a44e-04f85d81524a" width="300" height="250">

