# My projects

This repository contains source codes of projects I have been working on. 
<br>
- ## Applications
    - ### [**dataMate**](#datamate)
    - ### [**paperMate**](#papermate) <br>
- ## Data analysis
    - ### [**Skin pores topography**](./data_analysis/skin%20pores%20topography/analysis_skin%20pores%20topography.md)

## **dataMate** 
#### developed on 2020-08
#### [source code](./applications/dataMate.py)

<br>
dataMate is a program developed to assist input and analysis of data in clinical studies. <br>
The main screen of the program displays variables that need to be defined in order to prepare for a new study.<br>


<img src="https://github.com/hiwonis/projects/blob/main/applications/imgs/dM_main.png?raw=true" alt="Configuration settings window - paperMate" width="700"/> 

<br>
Once the variables are inputted according to the study's characteristics, right form of data input for the study is created in an Excel file. <br>

<img src="https://github.com/hiwonis/projects/blob/main/applications/imgs/pM_main_2.png?raw=true" alt="Configuration settings window - paperMate" width="700"/> 

<img src="https://github.com/hiwonis/projects/blob/main/applications/imgs/dM_dataform.png?raw=true" alt="Configuration settings window - paperMate" width="700"/> 

<br>
After the study is completed, the Excel file with inputted data can be loaded back into dataMate for analysis.<br>

<img src="https://github.com/hiwonis/projects/blob/main/applications/imgs/dM_analysis_2.png?raw=true" alt="Configuration settings window - paperMate" width="700"/> 

<br>
The data is sorted for easy understanding of trends over time, and results of tests for normality are displayed along with mean calculations. <br>

<img src="https://github.com/hiwonis/projects/blob/main/applications/imgs/dM_data%20analyzed.png?raw=true" alt="Configuration settings window - paperMate" width="700"/>

<br>
After conducting exploratory analysis like this, further analysis can be performed using specialized software such as SPSS. For utilizing in such softwares, dataMate also supports extraction sorted data in csv files. 
<br><br>

## **paperMate**
#### developed on 2023-03
#### [source code](./applications/paperMate.py)

<br>
paperMate is a program developed to automate paperwork. In some standardized tasks during paperwork, only a few parts need to be modified while the rest remains the same. By inputting only the information that needs to be modified, paperMate automatically completes the rest. By using Selenium library, paperMate can automate tasks on web-based systems related to paperwork as well.<br>

<img src="https://github.com/hiwonis/projects/blob/main/applications/imgs/pM_new%20task.png?raw=true" alt="Configuration settings window - paperMate" width="700"/>

<br>
The focus while developing this program was to make it user-friendly and prevent as many errors as possible. Therefore, instead of changing the source code of the program when some parts of paperwork should be modified, a "configuration settings' was added to allow users to modify the information directly within the program. <br>

<img src="https://github.com/hiwonis/projects/blob/main/applications/imgs/config_1.png?raw=true" alt="Configuration settings window - paperMate" width="700"/>

<br>
Also, to prevent erros due to incorrect information input by users, input validation was implemented to warn users in case of input somethings that can possibly cause errors.

<img src="https://github.com/hiwonis/projects/blob/main/applications/imgs/pM_new%20task_2.png?raw=true" alt="Configuration settings window - paperMate" width="700"/>

<img src="https://github.com/hiwonis/projects/blob/main/applications/imgs/pM_new%20task_login%20failed.png?raw=true" alt="Configuration settings window - paperMate" width="700"/>

