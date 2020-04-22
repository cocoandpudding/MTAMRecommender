# MTAMRecommender
For IJCAI2020<br/>
**please cite our paper if you use our codes.Thanks**
## Environment Settings
- python 3.6.3
- numpy 1.18.1
- pandas 0.20.3
- tensorflow 1.14.0
## Datasets
**we provide small datasets:MovieLens 1 Million (ml-1m)**<br/>
#### raw_data
- movies.dat
- ratings.dat
#### origin_data
- movielens.csv
#### training_testing_data
**rule :**<br/>
For the input behavior history of user u,[x_1,x_2,...x_{t-1},x_t]<br/>
**eg:[user_id,item_list,category_list,time_list,timelast_list,timenow_list,position_list,[target_id,target_category,target_time]**<br/>
- train_data.txt
- test_data.txt

