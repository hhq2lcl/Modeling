/********************************************************************************************/
/* Revised log																				*/
/* 2018-04-23, 新建, Zky 						   								    */
/*																							*/
/********************************************************************************************/
option compress = yes validvarname = any;
/*---input libname---*/
libname appraw odbc  datasrc=approval_nf;
libname approval "F:\A_offline_zky\kangyi\data_download\原表\approval";
libname res odbc  datasrc=res_nf;
libname credit "F:\A_offline_zky\kangyi\data_download\原表\credit";

/*---output libname---*/
libname orig "F:\share\model_development\02_DataSet\01_Original";

/*申请人信息汇总*/
data province;
set res.optionitem(where = (groupCode = "province"));
keep itemCode itemName_zh;
run;
data city;
set res.optionitem(where = (groupCode = "city"));
keep itemCode itemName_zh;
run;
data region;
set res.optionitem(where = (groupCode = "region"));
keep itemCode itemName_zh;
run;
data education;
set res.optionitem(where = (groupCode = "EDUCATION"));
keep itemCode itemName_zh;
run;
data gender;
set res.optionitem(where = (groupCode = "GENDER"));
keep itemCode itemName_zh;
run;
data marriage;
set res.optionitem(where = (groupCode = "MARRIAGE"));
keep itemCode itemName_zh;
run;
data house_property;
set res.optionitem(where = (groupCode = "PROPERTYTYPE"));
keep itemCode itemName_zh;
run;
data comp_type;
set res.optionitem(where = (groupCode = "COMPTYPE"));
keep itemCode itemName_zh;
run;
data position;
set res.optionitem(where = (groupCode = "POSITION"));
keep itemCode itemName_zh;
run;
data PERMANENT_TYPE;
set res.optionitem(where = (groupCode = "RegisteredType"));
keep itemCode itemName_zh;
run;

data apply_base;
set approval.apply_base(keep = apply_code PHONE1 ID_CARD_NO RESIDENCE_PROVINCE RESIDENCE_CITY RESIDENCE_DISTRICT PERMANENT_ADDR_PROVINCE PERMANENT_ADDR_CITY PERMANENT_TYPE
							PERMANENT_ADDR_DISTRICT LOCAL_RESCONDITION  EDUCATION MARRIAGE GENDER RESIDENCE_ADDRESS PERMANENT_ADDRESS CHILD_COUNT RESIDENCE_START_DATE);

/*RESIDENCE-现住址 PERMANENT-户籍地址*/
run;
proc sql;
create table base_info(drop= RESIDENCE_PROVINCE RESIDENCE_CITY RESIDENCE_DISTRICT PERMANENT_ADDR_PROVINCE PERMANENT_ADDR_CITY PERMANENT_ADDR_DISTRICT
 LOCAL_RESCONDITION EDUCATION MARRIAGE GENDER PERMANENT_TYPE) as
select a.*, b.itemName_zh as 居住省, c.itemName_zh as 居住市, d.itemName_zh as 居住区,
			e.itemName_zh as 户籍省, f.itemName_zh as 户籍市, g.itemName_zh as 户籍区,
			h.itemName_zh as 住房性质, i.itemName_zh as 教育程度, j.itemName_zh as 婚姻状况, k.itemName_zh as 性别,l.itemName_zh as 户口性质
from apply_base as a
left join province as b on a.RESIDENCE_PROVINCE = b.itemCode
left join city as c on a.RESIDENCE_CITY = c.itemCode
left join region as d on a.RESIDENCE_DISTRICT = d.itemCode
left join province as e on a.PERMANENT_ADDR_PROVINCE = e.itemCode
left join city as f on a.PERMANENT_ADDR_CITY = f.itemCode
left join region as g on a.PERMANENT_ADDR_DISTRICT = g.itemCode
left join house_property as h on a.LOCAL_RESCONDITION = h.itemCode
left join education as i on a.EDUCATION = i.itemCode
left join marriage as j on a.MARRIAGE = j.itemCode
left join gender as k on a.GENDER = k.itemCode
left join PERMANENT_TYPE as l on a.PERMANENT_TYPE = l.itemCode;
quit;
data apply_assets;
set approval.apply_assets;
if IS_HAS_HOURSE ="y" and IS_HAS_CAR="y"  then 财产信息 = "有房有车";
else if IS_HAS_HOURSE ="y" and IS_HAS_CAR="n"  then 财产信息 = "有房无车";
else if IS_HAS_HOURSE ="n" and IS_HAS_CAR="y"  then 财产信息 = "有车无房";
else  财产信息 = "无车无房";

keep APPLY_CODE housing_property IS_HAS_HOURSE IS_HAS_CAR 财产信息 IS_HAS_INSURANCE_POLICY HOURSE_COUNT CAR_COUNT IS_LIVE_WITH_PARENTS;
run;

proc sort data = apply_assets;by apply_code;run;

proc sql;
create table assets_info(drop=housing_property) as
select a.*, b.itemName_zh as 房产性质
from apply_assets as a
left join house_property as b on a.housing_property = b.itemCode
;
quit;
proc import out = ccoc datafile = "F:\share\Datamart\cs\wenjie\已补录CCOC码.xls" dbms = excel replace;
	getnames = yes;
run;

data industry;
set res.optionitem(where = (groupCode = "industry-1"));
keep itemCode itemName_zh;
run;
data CC;
set res.optionitem(where = (groupCode = "industry-2"));
keep itemCode itemName_zh;
run;
data OC;
set res.optionitem(where = (groupCode = "OC"));
keep itemCode itemName_zh;
run;

proc sql;
create table ccoc_info as
select a.apply_code, b.itemName_zh as 类别码, c.itemName_zh as CC码, d.itemName_zh as OC码
from ccoc as a
left join industry as b on a.industry_code = b.itemCode
left join CC as c on a.cc_code = c.itemCode
left join OC as d on a.oc_code = d.itemCode
;
quit;
/*---------------------补录的CCOC码 start-------------------------------------*/

data apply_emp;
set approval.apply_emp(keep = apply_code COMP_NAME position comp_type COMP_ADDR_PROVINCE COMP_ADDR_CITY COMP_ADDR_DISTRICT START_DATE_4_PRESENT_COMP
							CURRENT_INDUSTRY WORK_YEARS COMP_ADDRESS TITLE WORK_CHANGE_TIMES START_DATE_4_PRESENT_COMP);

							format 入职时间 yymmdd10.;
入职时间 = datepart(START_DATE_4_PRESENT_COMP);
							drop START_DATE_4_PRESENT_COMP;
run;
proc sql;
create table emp_info(drop= COMP_ADDR_PROVINCE COMP_ADDR_CITY COMP_ADDR_DISTRICT position comp_type START_DATE_4_PRESENT_COMP) as
select a.*, b.itemName_zh as 工作省, c.itemName_zh as 工作市, d.itemName_zh as 工作区,
			e.itemName_zh as 职级, f.itemName_zh as 单位性质
from apply_emp as a
left join province as b on a.COMP_ADDR_PROVINCE = b.itemCode
left join city as c on a.COMP_ADDR_CITY = c.itemCode
left join region as d on a.COMP_ADDR_DISTRICT = d.itemCode
left join position as e on a.position = e.itemCode
left join comp_type as f on a.comp_type = f.itemCode
;
quit;

data apply_ext_data;
set approval.apply_ext_data(keep = apply_code INDUSTRY_NAME CC_NAME OC_NAME);
run;



data apply_balance;
set approval.apply_balance(keep = apply_code yearly_income monthly_salary monthly_other_income monthly_expense 
							PUBLIC_FUNDS_RADICES SOCIAL_SECURITY_RADICES SALARY_PAY_WAY PAY_DAY);
run;


data salarypayway;
set res.optionitem(where = (groupCode = "SALARYPAYWAY"));
keep itemCode itemName_zh;
run;
proc sql;
create table balance_info(drop=SALARY_PAY_WAY) as select a.*,b.itemName_zh as 薪资发放方式   from  apply_balance as  a left join salarypayway as  b on a.SALARY_PAY_WAY =b.itemCode;quit;

data apply_info;
set approval.apply_info
(keep = apply_code name id_card_no DESIRED_LOAN_LIFE DESIRED_PRODUCT DESIRED_LOAN_AMOUNT BRANCH_NAME);
run;

data apply_time1;
set approval.act_opt_log(where = (task_Def_Name_ = "录入复核" and action_ = "COMPLETE")); /*action_必须是COMPLETE的才是进入审批的，JUMP的是复核时取消或拒绝*/
input_complete=1;/*action_必须是COMPLETE的才是进入审批的，JUMP的是复核时取消或拒绝*/
keep bussiness_key_ create_time_ input_complete;
rename bussiness_key_ = apply_code create_time_ = apply_time;
run;
proc sort data = apply_time1 dupout = a nodupkey; by apply_code apply_time; run;
proc sort data = apply_time1 nodupkey; by apply_code; run;
proc sort data = apply_info nodupkey; by apply_code; run;
data apply_time;
merge apply_time1(in = a) apply_info(in = b);
by apply_code;
if b;
run;
data apply;
set approval.act_opt_log(where = (task_Def_Name_ = "录入复核" and action_ = "COMPLETE")); /*action_必须是COMPLETE的才是进入审批的，JUMP的是复核时取消或拒绝*/
format 进件时间 YYMMDD10.;
进件时间=datepart(create_time_);
进件=1;
keep bussiness_key_ 进件时间 进件;
rename bussiness_key_ = apply_code ;
run;
proc sort data = apply dupout = a nodupkey; by apply_code 进件时间; run;
proc sort data = apply nodupkey; by apply_code; run;

proc sort data = credit.credit_report(keep=report_number id_card created_time) out = credit_report nodupkey; by report_number; run;

proc sql;
create table pboc_info as
select a.apply_code, b.*
from apply_time as a
inner join credit_report as b on a.id_card_no = b.id_card and datepart(a.apply_time) >= datepart(b.created_time)
;
quit;
proc sort data = pboc_info nodupkey; by apply_code descending created_time; run;
proc sort data = pboc_info  nodupkey; by apply_code; run;

data query_in3month;
set credit.credit_derived_data(drop = ID id_card CUSTOMER_CLASSIFICATION CREATED_USER_ID CREATED_USER_NAME CREATED_TIME 
								UPDATED_USER_ID UPDATED_USER_NAME UPDATED_TIME );

run;
proc sql;
create table pboc_info1 as
select a.*,b.* from pboc_info as a
left join query_in3month as b on a.REPORT_NUMBER=b.REPORT_NUMBER;
quit;

data contacts;
set approval.apply_contacts;
 if IS_KNOW_THAT ^="";
 if IS_KNOW_THAT ="0"  then IS_KNOW=0;
 else IS_KNOW=1;
if relation in ("190","191","192","193","FROTHER")  and IS_KNOW=1 then 直系亲属=1;else 直系亲属=0;
if relation in ("187","MR001") and IS_KNOW=1 then 配偶=1;else 配偶=0;
if relation in ("201") and IS_KNOW=1 then 工作证明人=1;else 工作证明人=0;
if relation in ("156","157","158","159") and IS_KNOW=1 then 其他=1;else 其他=0;

 run;
proc sql;
create table cc as select apply_code,sum(IS_KNOW) as know_num,sum(直系亲属) as fam_know, sum(配偶) as mate_know , sum(工作证明人) as job_know,
sum(其他) as oth_know from contacts group by apply_code;quit;

proc sort data = base_info nodupkey; by apply_code; run;
proc sort data = emp_info nodupkey; by apply_code; run;
proc sort data = apply_ext_data nodupkey; by apply_code; run;
proc sort data = assets_info nodupkey; by apply_code; run;
proc sort data = ccoc_info nodupkey; by apply_code; run;
proc sort data = balance_info nodupkey ; by apply_code;run;
proc sort data = apply_info nodupkey;by apply_code;run;
proc sort data = pboc_info1 nodupkey;by apply_code;run;
proc sort data = approval.credit_score out =credit_score(keep = group_Level apply_code)
 nodupkey;by apply_code;run;

data customer_info;
merge apply(in =a )  base_info(in = b) emp_info(in = c) apply_ext_data(in = d) 
  assets_info(in = e) ccoc_info(in = f)  balance_info(in = g)  apply_info(in = h) pboc_info1(in=i) credit_score(in=j) ;
by apply_code;
if a;
/*年龄计算 ID_CARD_NO 取出生年月日计算-实足年龄*/
format age 10.;
format birthdate yymmdd10.;
birth_year=substr(ID_CARD_NO,7,4)+0;
birth_mon=substr(ID_CARD_NO,11,2)+0;
birth_day=substr(ID_CARD_NO,13,2)+0;
birthdate=mdy(birth_mon,birth_day,birth_year);
age=Intck('year',birthdate,进件时间);
format RESIDENCE_START_DATE1 yymmdd10.;
RESIDENCE_START_DATE1=datepart(RESIDENCE_START_DATE);
LOCAL_RES_YEARS=Intck('year',RESIDENCE_START_DATE1,进件时间);
drop birth_mon birth_day birth_year;
if 住房性质 = "" then 住房性质 = 房产性质;
if INDUSTRY_NAME = "" then INDUSTRY_NAME = 类别码;
if fund_month=. then fund_month=PUBLIC_FUNDS_RADICES;
if social_security_month=. then social_security_month=SOCIAL_SECURITY_RADICES;
if fund_month < social_security_month  or fund_month=.  then  fund_month = social_security_month;

if CC_NAME = "" then CC_NAME = CC码;
if OC_NAME = "" then OC_NAME = OC码;
format 外地标签 $20.;
if 户籍市="" or 工作市="" then 外地标签="数据缺失";
else if 户籍市^=工作市 then 外地标签="外地";
else 外地标签="本地";
if  INDUSTRY_NAME in ("酒店","餐饮","其他行业","一般制造业") then job_manu=1 ;
if kindex(INDUSTRY_NAME,"制造业") and kindex(CC_name ,"制造业")  then job_manu =1 ;

if 户籍省 in ("河南省","江西省","湖北省","湖南省")  then do; area = "华中";area1=0;end ;
if 户籍省 in ("辽宁省","黑龙江省","吉林省")  then do;area1=1;area = "东北";end;
if 户籍省 in ("山西省","山东省","北京市","天津市","河北省")  then do; area1 =2 ;area = "华北"; end ;
if 户籍省 in ("上海市","浙江省","江苏省","安徽省")  then do; area1 =3;area = "华东";end;
if 户籍省 in ("广东省","广西壮族自治区","海南省","福建省")  then do;area1=4; area = "华南";end;
if 户籍省 in ("四川省","重庆市","昆明市","贵州省","云南省","西藏自治区")  then do; area1 =5; area = "西南";end;
if 户籍省 in ("陕西省","宁夏回族自治区","青海省","甘肃省","新疆维吾尔自治区")  then do; area1 =6; area = "西北";end;
if 户籍省 ="内蒙古自治区" then do; area1 =7;area = "内蒙古自治区";end ;

drop  类别码 CC码 OC码 birthdate 进件时间  进件  COMP_NAME CC_NAME OC_NAME COMP_ADDRESS RESIDENCE_ADDRESS PERMANENT_ADDRESS PHONE1 PAY_DAY
	 居住区 户籍区 工作区 居住省 居住市 工作省 工作市 户籍省 户籍市 CURRENT_INDUSTRY TITLE PUBLIC_FUNDS_RADICES SOCIAL_SECURITY_RADICES RESIDENCE_START_DATE1 RESIDENCE_START_DATE yearly_income hourse_count
 CAR_COUNT social_security_month ;

/*{"住房性质":{"亲属住房":0,"公司宿舍":1,"公积金按揭购房":2,"其他":3,"商业按揭房":4,"无按揭购房":5,"租用":6,"自建房":7},*/
/*        "外地标签":{"本地":0,"外地":1},"薪资发放方式":{"均有":0,"打卡":1,"现金":2,"其他":3,"银行代发":3},*/
/*        "财产信息":{"无车无房":0,"有车无房":1,"有房无车":2,"有车有房":3},"IS_LIVE_WITH_PARENTS":{"y":0,"n":1},*/
/*        "IS_HAS_CAR":{"y":0,"n":1},"IS_HAS_INSURANCE_POLICY":{"y":0,"n":1},"IS_HAS_HOURSE":{"y":0,"n":1},*/
/*         "教育程度":{"研究生及其以上":0,"大学本科":1,"专科":2,"中专":3,"高中":4,"初中":5,"小学":6},*/
/*        "婚姻状态":{"未婚":0,"已婚":1,"丧偶":2,"离异":3},*/
/*        "职级":{"一般正式员工": 0,"一般管理人员": 1,"中级管理人员": 2,"派遣员工": 3,"负责人": 4,"非正式员工": 5,"高级管理人员": 6},*/
/*        "单位性质":{"个体":0,"合资企业":1,"国有股份":2,"外资企业":3,"机关事业单位":4,"民营企业":5,"社会团体":6,"私营企业":7}*/
/*教育分类*/
if  教育程度="硕士及其以上" then  EDUCATION=0;
if 教育程度="大学本科" then  EDUCATION=1;
if  教育程度="专科" then EDUCATION=2;
if  教育程度="中专" then EDUCATION=3;
if 教育程度="高中" then EDUCATION=4;
if  教育程度="初中" then EDUCATION=5;
if 教育程度="小学" then  EDUCATION=6;
drop 教育程度;
/*婚姻状况*/
if 婚姻状况="未婚" then  MARRIAGE=0;
if 婚姻状况="已婚" then MARRIAGE=1;
if 婚姻状况="丧偶" then MARRIAGE=2;
if 婚姻状况="离异" then  MARRIAGE=3;
drop 婚姻状况;

/*性别*/
if  性别="男" then GENDER1=0 ; 
if 性别="女" then GENDER1=1 ; 
drop 性别;

/*房产信息*/
if IS_HAS_HOURSE="y" then IS_HAS_HOURSE1=0;
else if IS_HAS_HOURSE="n" then IS_HAS_HOURSE1=1;
drop IS_HAS_HOURSE;

/*if IS_HAS_CAR="y" then IS_HAS_CAR1=0;*/
/*else if IS_HAS_CAR="n" then IS_HAS_CAR1=1;*/
drop IS_HAS_CAR;


/*if IS_LIVE_WITH_PARENTS="y"  then IS_LIVE_WITH_PARENTS1=0;*/
/*else if IS_LIVE_WITH_PARENTS="n" then IS_LIVE_WITH_PARENTS1=1;*/
drop IS_LIVE_WITH_PARENTS;


/*if IS_HAS_INSURANCE_POLICY="y" then IS_HAS_INSUR=0;*/
/*else IS_HAS_INSUR=1;*/
drop IS_HAS_INSURANCE_POLICY;

/*户口类型*/
if 户口性质="本地户籍" then do; PERMANENT_TYPE1=0;;end;
else if 户口性质="本地农村"  then do; PERMANENT_TYPE1=1;;end;
else if 户口性质="外地户籍" then do; PERMANENT_TYPE1=2;;end;
else PERMANENT_TYPE1=3;
drop 户口性质;

/*工资发放路径*/
/*if 薪资发放方式="现金" then SALARY_PAY_WAY1=2;*/
/*if 薪资发放方式="打卡" then SALARY_PAY_WAY1=1;*/
/*if 薪资发放方式="银行代发" then SALARY_PAY_WAY1=3;*/
/*if 薪资发放方式="其他" then SALARY_PAY_WAY1=3;*/
/*if 薪资发放方式="均有" then SALARY_PAY_WAY1=0;*/
drop 薪资发放方式;
/*供养人数*/

/*房产性质*/
if 房产性质="公积金按揭购房"  then LOCAL_RESCONDITION_G =0;
else if 房产性质="公司宿舍"  then LOCAL_RESCONDITION_G =1;
else if 房产性质="亲属住房"  then LOCAL_RESCONDITION_G =2;
else if 房产性质="商业按揭房"  then LOCAL_RESCONDITION_G =3;
else if 房产性质="无按揭购房"  then LOCAL_RESCONDITION_G =4;
else if 房产性质="自建房"  then LOCAL_RESCONDITION_G =5;
else if 房产性质="租用"  then LOCAL_RESCONDITION_G =6;
else if 房产性质="其他"  then LOCAL_RESCONDITION_G =7;
drop 房产性质 住房性质 入职时间;

/*职级*/
if 职级= "非正式员工" then position_G =5;
else if 职级= "负责人" then position_G =4;
else if 职级= "高级管理人员" then position_G =6;
else if 职级= "派遣员工" then position_G =3;
else if 职级= "一般管理人员" then position_G =1;
else if 职级= "一般正式员工" then position_G =0;
else if 职级= "中级管理人员" then position_G =2;
else position_G =7;
drop 职级;

/*单位性质*/
if 单位性质 ="个体" then comp_type=0;
else if 单位性质 ="国有股份" then comp_type=2;
else if 单位性质 ="合资企业" then comp_type=1;
else if 单位性质 ="机关事业单位" then comp_type=4;
else if 单位性质 ="民营企业" then comp_type=5;
else if 单位性质 ="社会团体" then comp_type=6;
else if 单位性质 ="私营企业" then comp_type=7;
else if 单位性质 ="外资企业" then comp_type=3;
drop 单位性质;


/*外地标签*/
if  外地标签="外地"  then  nonlocal=0;
else if 外地标签="本地"  then  nonlocal=1;
drop 外地标签;

/*财产信息*/
/*if 财产信息 = "有房有车" then asset = 3;*/
/*else if 财产信息 = "有房无车" then asset =2;*/
/*else if 财产信息 = "有车无房" then asset = 1;*/
/*else if 财产信息 = "无车无房" then asset = 0;*/
drop 财产信息 INDUSTRY_NAME;

/*评分*/
if group_Level ="A" then group_Level_g =0;
if group_Level ="B" then group_Level_g =1;
if group_Level ="C" then group_Level_g =2;
if group_Level ="D" then group_Level_g =3;
if group_Level ="E" then group_Level_g =4;
if group_Level ="F" then group_Level_g =5;
DROP GROUP_LEVEL;

if kindex(DESIRED_PRODUCT,"El") then DESIRED_PRODUCT1=0;
else DESIRED_PRODUCT1=1;

drop DESIRED_PRODUCT;

if DESIRED_LOAN_LIFE ="344"  then loan_live = 24;
if DESIRED_LOAN_LIFE ="345"  then loan_live = 36;

if kindex(BRANCH_NAME,"上海") or kindex(BRANCH_NAME,"赤峰")or kindex(BRANCH_NAME,"怀化")or kindex(BRANCH_NAME,"厦门") then branch_=2;
else if kindex(BRANCH_NAME,"业务中心")  then branch_=1;
else branch_=0;
drop BRANCH_NAME area;
run;



data orig.customer_info;
set customer_info;
run;

