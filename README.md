# Most Likely Heteroscedastic Gaussian Process Regression  
ICML2007で発表された「Most Likely Heteroscedastic Gaussian Process Regression」を読み，提案手法を実装しました．  
論文URL：http://people.csail.mit.edu/kersting/papers/kersting07icml_mlHetGP.pdf  

## Heteroscedastic Gaussian Process Regressionとは  
HGPRは邦訳だと異分散ガウス過程回帰と呼ばれるアルゴリズムです．
y=f(x)の関数が存在する時，xの値によってyの従う確率分布の分散が変化するような関数を表します．
例えば，論文中ではいくつかのテスト関数を用いて検証しているが，その一つが Yuan and Wahba (2004)で用いられた関数であり，式は以下のように定義されます．  
  
![Yuan_and_Wahba](https://github.com/sylvesterml/Most-Likely-HGPR/blob/master/pictures/Yuan_and_Wahba.png)  
  
また，この関数を図示すると下図のようになります．
ここで，赤線はyが従う確率分布の平均値μ(x)を表し，またyが従う確率分布の標準偏差をσ(x)とすると，シアンで塗られた領域は内側から順に，σ(x)，2σ(x)，3σ(x)をそれぞれ表し，青点はこのテスト関数を元にサンプリングしたデータ点を表す．  
![train](https://github.com/sylvesterml/Most-Likely-HGPR/blob/master/pictures/train.png)  
通常のGPRではyの分散は定数であり，xの値に対して不変であると仮定するため，このような関数に対して適用することができません．
そこで，このような関数の回帰を行うためにHGPRが提案されました．
HGPRでは，GPRにおけるノイズ行列をGPRによって導出し，導出されたノイズ行列を用いて再びGPRを行うことで回帰を行います．  

## Most Likely Heteroscedastic Gaussian Process Regressionとは  
HGPRの問題点として，GPRの計算量自体がO(n<sup>3</sup>)（n:データ数）であるため，愚直にGPRを2回繰り返すとかなり計算量が大きくなってしまうというものがあります．
そこで，ノイズ行列を導出するためのGPRの過程を何かしらの手法で近似することで高速化が図られてきました．
今回の提案手法であるMost Likely Heteroscedastic Gaussian Process Regressionは，この近似手法としてEMアルゴリズムを採用しています．
これらの一連の流れは以下のアルゴリズムで表されます．
変数等の定義は論文を参照してください．  
  
![hgpr_algorithm](https://github.com/sylvesterml/Most-Likely-HGPR/blob/master/pictures/hgpr_algorithm.png)  
  

## Yuan and Wahba (2004)のテスト関数による実験  
前述したYuan and Wahba (2004)で用いられたテスト関数を用いて実験を行います．  
実験結果は下図のようになります．
ここで，赤線はyが従う確率分布の平均値μ(x)を表し，またyが従う確率分布の標準偏差をσ(x)とすると，ピンク線は内側から順に，σ(x)，2σ(x)，3σ(x)をそれぞれ表し，緑点はこのテスト関数を元にサンプリングしたデータ点を表します． 
また，青線はサンプリングしたデータを元に予測したyが従う確率分布の平均値μ(x)\*を表し，予測したyが従う確率分布の標準偏差をσ(x)\*とすると，シアンで塗られた領域は内側から順に，σ(x)\*，2σ(x)\*，3σ(x)\*をそれぞれ表します．  
![hgpr_result](https://github.com/sylvesterml/Most-Likely-HGPR/blob/master/pictures/Yuan_Wahba_data30.png)  
