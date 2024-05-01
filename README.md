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
> - <img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/a15ec870-982e-4213-9699-0a3180afd0f4/to/image" width="500" height="400" alt="Benford's Law with physics constants">
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
> 
> So, we suggest that, if a company's first digit frequency differs a lot from the Benford's Law, it's more likely that the company has made financial fraud.
>
> We will use different machine learning methods to prove our thoughts, where independent varaibles are the difference between a company's first digit frequency, and the dependent variable is whether the company has made financial fraud.

> In previous literature, 
>
> - [Benford's Law is used to detect fraud of credit card transactions in social media](https://ieeexplore.ieee.org/abstract/document/9016804); 
>
> - [Bao[2020] used machine learning methods (without using Benford's Law) on original numbers of three financial statements to detect accounting fraud for U.S. stocks](https://onlinelibrary.wiley.com/doi/full/10.1111/1475-679X.12292)
>
> To the best of my knowledge, our method, using Benford's Law on all positive original numbers of three financial statments to detect financial fraud for public traded companiess, has not been conducted before. 


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
> y is determinde by the auditor's opinion on the financial statements:
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
> We get the ~300 positive numbers (or financial accounts)  of each financial statement through WIND API

> Then we compute the first digit frequency, and the distribution is as follows:
> 
> <img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/a0fffa48-7e74-49de-a35a-a62f9bcc6e88/to/image" width="600" height="400" alt="Samples_Benford's Law">

> We get auditor's opinion on financial statements through WIND EXCEL Plugger

### 4.5 Data processing
> $X_i$ describes the difference between real frequency and Benford frequency:
>
> $$ X_i = \frac{{\rm{frequency \ \ of \ \ beginning \ \ with \ \ digit \ \ i}}}{{\rm{Benford \ \  frequency\ \ [i]}}} $$
>
> We use the largest $X_i$ within the 5 years (2015~2019).

> Obviously, the first digit frequency of Fraud Group differs more from Benford Frequency than that of No Fraud Group:
![output](https://github.com/pkujackfeng/MLF_final_project/assets/90912432/52926052-5bc2-416f-9b08-26e2b535254b)


> A big issue is that the dataset is imbalanced, there are too few samples with y= 0.
>
> So within the trainning set we do under-sampling using **RandomUnderSampler**：
>
> we randomly drop some negative samples (y=0) until the number of negative samples equal to the number of positive samples. 
>
> - Before RandomUnderSampler, 36/279=12% samples are y=1.
> - After RandomUnderSampler, 36/72=50% samples are y=1.

### 4.6 The Processed Data
> ![图片](https://github.com/pkujackfeng/MLF_final_project/assets/90912432/0d4d7635-5d1d-4448-a347-d9d282dec28e)

## 5. Model Results
> Our task is to tell whether there is financial fraud, so we care about two index: 
> - 1) recall
>
> - 2) auc

> Logistic Regression and SVM get the best result, while Decision Tress gives the worst result:
> 
|  | Logistic Regression | MLPClassfier | SVM | Decision Tree | Random Forest |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Recall | 69% | 62% | 69% | 44% | 62% |
| AUC | 63% | 62% | 63% | 56% | 63% |
> ![img](https://github.com/pkujackfeng/MLF_final_project/assets/90912432/25a2311d-e820-431e-b17a-93186ec0f3ad)

### 5.1 Logistic Regression；Recall = 69%, AUC = 63%
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/a4814667-f53f-46ef-bfa3-68261670b547" width="400" height="250">
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/3b158be1-5c48-4b3b-9d8b-e3abaecc9d25" width="300" height="250">

### 5.2 MLPClassifier: Recall = 62%, AUC = 62%
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/6464ef17-4837-4094-bd9c-d7ddb48f683b" width="400" height="250">
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/f50a5742-d56c-401c-8dbc-7a0ac567c021" width="300" height="250">

### 5.3 SVM: Recall = 69%, AUC = 63%
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/2a4abf62-96b9-4197-a0df-cecaf0d1b1f5" width="400" height="250">
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/0f1ff371-1256-488d-9181-ef620d65b56d" width="300" height="250">

### 5.4 Decision Tree:Recall = 44%, AUC = 56%
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/79f38601-8384-4bf2-8dd9-562adf82d472" width="400" height="250">
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/fbced0f8-3437-4e9d-bc60-29c5ca24a0ea" width="300" height="250">

### 5.5 Random Forest: Recall = 62%, AUC = 63%
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/64f0c29e-2e41-4f13-bf71-5c74f54270c4" width="400" height="250">
<img src="https://github.com/pkujackfeng/MLF_final_project/assets/90912432/207f27e7-4db4-4e21-b011-2dabde8e983e" width="300" height="250">

