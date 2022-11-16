
# 파편화된 데이터의 적극 활용을 위한 시계열 기반 통합 플랫폼 기술 개발

- 참여 기관명 : 광운대학교, 신경공학 및 인공지능 연구실(NeuroAI Lab)

- 연구책임자 : 최영석 교수

- 개발자 정보
  - 김채민 연구원
  - Email : k9is@kw.ac.kr 

- 참여 기관 실무 담당자 및 작성자 정보
  - 이현규 연구원
  - Email : skgusrb12@kw.ac.kr 

## 1. Introduction

- **엔트로피(Entropy) 기법을 활용한 시계열 데이터의 비정규성(irregularity) 및 randomness 정량화 기술 개발**

- **개발 기술 요약** 

|제안 기법|기능|활용 예제|
|:--------:|:-------:|:------:|
|멀티 스케일 누적 잔여 분산 엔트로피 (Multiscale Cumulative Residual Dispersion Entropy, MCRDE)|  시계열 데이터 기반의 정상/비정상(이상 데이터) 특징 추출 및 상황 감지| 심박변이율의 복잡도를 다양한 시간의 스케일에서 분석하여 울혈성 심부전 환자와 정상인 데이터를 구분하기 위함



## 2. Sample Dataset (Input Data Formats)

- PhysioNet(https://physionet.org)에서 제공되는 울혈성 심부전증(BIDMC CHF, CHFDB) 환자 데이터가 활용되었으며, 입력 데이터는 실험자 수 x 데이터 길이(S x N)로 구성됨.

- S : 1 x N 크기의 시계열 데이터의 개수
- N : 데이터 길이

*   in main.py
	```python
	N : 데이터 길이 (default: 1000)
	m : 데이터 차원 수 (default: 3)
	c : 클래스 수 (default: 6)		                
	tau : 지연 인자 (default: 1)
	scale : 스케일 수 (default: 25) 
	```
	- c : 각각의 데이터에 대해 1에서 c로 mapping
	- m :  1에서 c까지의 정수로 맵핑된 데이터를 m개의 분산 패턴로 표현
	- tau : 분산 패턴을 표시하는 간격
	- scale : 새로운 시계열 데이터 생성을 위한 데이터 추출 개수 
	
	
## 3. Usage

### `Main.py` 



- **Output Data Formats**
	```python
	mcrde_chf     = np.zeros((n_s, scale))
	mcrde_healthy = np.zeros((n_s, scale)) 
	```
  - n_s : 데이터의 수
  - scale : 해당 스케일 수
  -  example) mcrde_chf(3,15): 3번째 CHF환자의 scale 15에서의 MCRDE를 통해 정량화된 수치

	
	
- **Multiscale Cumulative Residual Dispersion Entropy (MCRDE)**

  ```python
  for i in range(n_s):  
    mcrde_chf[i] = MCRDE(RRIs_CHF_1000[i, :N], m, c, tau, scale)  
    mcrde_healthy[i] = MCRDE(RRIs_HEALTHY_1000[i, :N], m, c, tau, scale)  
  
  avg_mcrde_chf = np.mean(mcrde_chf, axis=0)  
  avg_mcrde_healthy = np.mean(mcrde_healthy, axis=0)  
    
  std_mcrde_chf = np.std(mcrde_chf, axis=0)  
  std_mcrde_healthy = np.std(mcrde_healthy, axis=0)
  ```
  - avg_mcrde_chf : 전체 울혈성 심부전증 환자 데이터에 대한 MCRDE의 평균값
  - avg_mcrde_healthy : 전체 울혈성 심부전증 환자 데이터에 대한 MCRDE의 표준편차
  - std_mcrde_chf : 전체 정상인 데이터에 대한 MCRDE의 평균값
  - std_mcrde_healthy : 전체 정상인 데이터에 대한 MCRDE의 표준편차

  

-	**결과 저장 (MCRDE의 평균 및 표준편차)**
	
	```python 
	write_txt(avg_mcrde_chf_path, avg_mcrde_healthy_path,
	         std_mcrde_chf_path, std_mcrde_healthy_path,
	         avg_mcrde_chf, avg_mcrde_healthy,
	         std_mcrde_chf, std_mcrde_healthy)
	```
	-  example) 전체 울혈성 심부전증 환자 데이터에 대한 MCRDE의 평균값 (path : results/avg_mcrde_chf.txt)

	   <img src="https://github.com/piggymouse/MCRDE-report/blob/main/avg,std/avg_mcrde_chf.png?raw=true" width=300> 



## 4. Results

-  **Figure parameter**

	  ```python
	if show_fig == True:
	    Entropy = plot_Entropy(subject,plt_color,
	                            avg_mcrde_chf,
	                            avg_mcrde_healthy,
	                            std_mcrde_chf,
	                            std_mcrde_healthy,
	                            scale,
	                            pCHF_HT)
	```
	
	 - subject: 그래프의 legend
	 - plt_color: 그래프의 색상
	 - avg_mcrde: MCRDE 평균 값
	 - std_mcrde: MCRDE 표준 편차 값
	 - scale: 스케일
	 - pCHF_HT: CHF환자와 정상인 사이의 p value
	
		
- **MCRDE 결과 그래프**


![Figure_1](https://user-images.githubusercontent.com/51149957/155088721-e4bffccf-3acc-4a4b-b237-9da0ffc5f023.jpeg)


	* 데이터의 길이(N): 1000
	* x 축: Scale factor 
	* y 축: Entropy Value 
	
   * MCRDE의 평균: 점으로 표시
   * MCRDE의 표준편차: 에러바로 표시
   *  에러바 위의 * 표시는 해당 되는 스케일에서 두 데이터(CHF, HEALTHY)가 서로 독립적이며 서로 다른 데이터라는 것을 의미함
   *  MCRDE 그래프를 통해서, 스케일 1을 제외한 모든 스케일에서  CHF subjects와 HEALTHY subjects의 구분이 가능
       
    



