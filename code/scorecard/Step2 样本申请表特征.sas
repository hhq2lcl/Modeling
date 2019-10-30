/********************************************************************************************/
/* Revised log																				*/
/* 2018-04-23, �½�, Zky 						   								    */
/*																							*/
/********************************************************************************************/
option compress = yes validvarname = any;
/*---input libname---*/
libname appraw odbc  datasrc=approval_nf;
libname approval "F:\A_offline_zky\kangyi\data_download\ԭ��\approval";
libname res odbc  datasrc=res_nf;
libname credit "F:\A_offline_zky\kangyi\data_download\ԭ��\credit";

/*---output libname---*/
libname orig "F:\share\model_development\02_DataSet\01_Original";

/*��������Ϣ����*/
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

/*RESIDENCE-��סַ PERMANENT-������ַ*/
run;
proc sql;
create table base_info(drop= RESIDENCE_PROVINCE RESIDENCE_CITY RESIDENCE_DISTRICT PERMANENT_ADDR_PROVINCE PERMANENT_ADDR_CITY PERMANENT_ADDR_DISTRICT
 LOCAL_RESCONDITION EDUCATION MARRIAGE GENDER PERMANENT_TYPE) as
select a.*, b.itemName_zh as ��סʡ, c.itemName_zh as ��ס��, d.itemName_zh as ��ס��,
			e.itemName_zh as ����ʡ, f.itemName_zh as ������, g.itemName_zh as ������,
			h.itemName_zh as ס������, i.itemName_zh as �����̶�, j.itemName_zh as ����״��, k.itemName_zh as �Ա�,l.itemName_zh as ��������
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
if IS_HAS_HOURSE ="y" and IS_HAS_CAR="y"  then �Ʋ���Ϣ = "�з��г�";
else if IS_HAS_HOURSE ="y" and IS_HAS_CAR="n"  then �Ʋ���Ϣ = "�з��޳�";
else if IS_HAS_HOURSE ="n" and IS_HAS_CAR="y"  then �Ʋ���Ϣ = "�г��޷�";
else  �Ʋ���Ϣ = "�޳��޷�";

keep APPLY_CODE housing_property IS_HAS_HOURSE IS_HAS_CAR �Ʋ���Ϣ IS_HAS_INSURANCE_POLICY HOURSE_COUNT CAR_COUNT IS_LIVE_WITH_PARENTS;
run;

proc sort data = apply_assets;by apply_code;run;

proc sql;
create table assets_info(drop=housing_property) as
select a.*, b.itemName_zh as ��������
from apply_assets as a
left join house_property as b on a.housing_property = b.itemCode
;
quit;
proc import out = ccoc datafile = "F:\share\Datamart\cs\wenjie\�Ѳ�¼CCOC��.xls" dbms = excel replace;
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
select a.apply_code, b.itemName_zh as �����, c.itemName_zh as CC��, d.itemName_zh as OC��
from ccoc as a
left join industry as b on a.industry_code = b.itemCode
left join CC as c on a.cc_code = c.itemCode
left join OC as d on a.oc_code = d.itemCode
;
quit;
/*---------------------��¼��CCOC�� start-------------------------------------*/

data apply_emp;
set approval.apply_emp(keep = apply_code COMP_NAME position comp_type COMP_ADDR_PROVINCE COMP_ADDR_CITY COMP_ADDR_DISTRICT START_DATE_4_PRESENT_COMP
							CURRENT_INDUSTRY WORK_YEARS COMP_ADDRESS TITLE WORK_CHANGE_TIMES START_DATE_4_PRESENT_COMP);

							format ��ְʱ�� yymmdd10.;
��ְʱ�� = datepart(START_DATE_4_PRESENT_COMP);
							drop START_DATE_4_PRESENT_COMP;
run;
proc sql;
create table emp_info(drop= COMP_ADDR_PROVINCE COMP_ADDR_CITY COMP_ADDR_DISTRICT position comp_type START_DATE_4_PRESENT_COMP) as
select a.*, b.itemName_zh as ����ʡ, c.itemName_zh as ������, d.itemName_zh as ������,
			e.itemName_zh as ְ��, f.itemName_zh as ��λ����
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
create table balance_info(drop=SALARY_PAY_WAY) as select a.*,b.itemName_zh as н�ʷ��ŷ�ʽ   from  apply_balance as  a left join salarypayway as  b on a.SALARY_PAY_WAY =b.itemCode;quit;

data apply_info;
set approval.apply_info
(keep = apply_code name id_card_no DESIRED_LOAN_LIFE DESIRED_PRODUCT DESIRED_LOAN_AMOUNT BRANCH_NAME);
run;

data apply_time1;
set approval.act_opt_log(where = (task_Def_Name_ = "¼�븴��" and action_ = "COMPLETE")); /*action_������COMPLETE�Ĳ��ǽ��������ģ�JUMP���Ǹ���ʱȡ����ܾ�*/
input_complete=1;/*action_������COMPLETE�Ĳ��ǽ��������ģ�JUMP���Ǹ���ʱȡ����ܾ�*/
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
set approval.act_opt_log(where = (task_Def_Name_ = "¼�븴��" and action_ = "COMPLETE")); /*action_������COMPLETE�Ĳ��ǽ��������ģ�JUMP���Ǹ���ʱȡ����ܾ�*/
format ����ʱ�� YYMMDD10.;
����ʱ��=datepart(create_time_);
����=1;
keep bussiness_key_ ����ʱ�� ����;
rename bussiness_key_ = apply_code ;
run;
proc sort data = apply dupout = a nodupkey; by apply_code ����ʱ��; run;
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
if relation in ("190","191","192","193","FROTHER")  and IS_KNOW=1 then ֱϵ����=1;else ֱϵ����=0;
if relation in ("187","MR001") and IS_KNOW=1 then ��ż=1;else ��ż=0;
if relation in ("201") and IS_KNOW=1 then ����֤����=1;else ����֤����=0;
if relation in ("156","157","158","159") and IS_KNOW=1 then ����=1;else ����=0;

 run;
proc sql;
create table cc as select apply_code,sum(IS_KNOW) as know_num,sum(ֱϵ����) as fam_know, sum(��ż) as mate_know , sum(����֤����) as job_know,
sum(����) as oth_know from contacts group by apply_code;quit;

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
/*������� ID_CARD_NO ȡ���������ռ���-ʵ������*/
format age 10.;
format birthdate yymmdd10.;
birth_year=substr(ID_CARD_NO,7,4)+0;
birth_mon=substr(ID_CARD_NO,11,2)+0;
birth_day=substr(ID_CARD_NO,13,2)+0;
birthdate=mdy(birth_mon,birth_day,birth_year);
age=Intck('year',birthdate,����ʱ��);
format RESIDENCE_START_DATE1 yymmdd10.;
RESIDENCE_START_DATE1=datepart(RESIDENCE_START_DATE);
LOCAL_RES_YEARS=Intck('year',RESIDENCE_START_DATE1,����ʱ��);
drop birth_mon birth_day birth_year;
if ס������ = "" then ס������ = ��������;
if INDUSTRY_NAME = "" then INDUSTRY_NAME = �����;
if fund_month=. then fund_month=PUBLIC_FUNDS_RADICES;
if social_security_month=. then social_security_month=SOCIAL_SECURITY_RADICES;
if fund_month < social_security_month  or fund_month=.  then  fund_month = social_security_month;

if CC_NAME = "" then CC_NAME = CC��;
if OC_NAME = "" then OC_NAME = OC��;
format ��ر�ǩ $20.;
if ������="" or ������="" then ��ر�ǩ="����ȱʧ";
else if ������^=������ then ��ر�ǩ="���";
else ��ر�ǩ="����";
if  INDUSTRY_NAME in ("�Ƶ�","����","������ҵ","һ������ҵ") then job_manu=1 ;
if kindex(INDUSTRY_NAME,"����ҵ") and kindex(CC_name ,"����ҵ")  then job_manu =1 ;

if ����ʡ in ("����ʡ","����ʡ","����ʡ","����ʡ")  then do; area = "����";area1=0;end ;
if ����ʡ in ("����ʡ","������ʡ","����ʡ")  then do;area1=1;area = "����";end;
if ����ʡ in ("ɽ��ʡ","ɽ��ʡ","������","�����","�ӱ�ʡ")  then do; area1 =2 ;area = "����"; end ;
if ����ʡ in ("�Ϻ���","�㽭ʡ","����ʡ","����ʡ")  then do; area1 =3;area = "����";end;
if ����ʡ in ("�㶫ʡ","����׳��������","����ʡ","����ʡ")  then do;area1=4; area = "����";end;
if ����ʡ in ("�Ĵ�ʡ","������","������","����ʡ","����ʡ","����������")  then do; area1 =5; area = "����";end;
if ����ʡ in ("����ʡ","���Ļ���������","�ຣʡ","����ʡ","�½�ά���������")  then do; area1 =6; area = "����";end;
if ����ʡ ="���ɹ�������" then do; area1 =7;area = "���ɹ�������";end ;

drop  ����� CC�� OC�� birthdate ����ʱ��  ����  COMP_NAME CC_NAME OC_NAME COMP_ADDRESS RESIDENCE_ADDRESS PERMANENT_ADDRESS PHONE1 PAY_DAY
	 ��ס�� ������ ������ ��סʡ ��ס�� ����ʡ ������ ����ʡ ������ CURRENT_INDUSTRY TITLE PUBLIC_FUNDS_RADICES SOCIAL_SECURITY_RADICES RESIDENCE_START_DATE1 RESIDENCE_START_DATE yearly_income hourse_count
 CAR_COUNT social_security_month ;

/*{"ס������":{"����ס��":0,"��˾����":1,"�����𰴽ҹ���":2,"����":3,"��ҵ���ҷ�":4,"�ް��ҹ���":5,"����":6,"�Խ���":7},*/
/*        "��ر�ǩ":{"����":0,"���":1},"н�ʷ��ŷ�ʽ":{"����":0,"��":1,"�ֽ�":2,"����":3,"���д���":3},*/
/*        "�Ʋ���Ϣ":{"�޳��޷�":0,"�г��޷�":1,"�з��޳�":2,"�г��з�":3},"IS_LIVE_WITH_PARENTS":{"y":0,"n":1},*/
/*        "IS_HAS_CAR":{"y":0,"n":1},"IS_HAS_INSURANCE_POLICY":{"y":0,"n":1},"IS_HAS_HOURSE":{"y":0,"n":1},*/
/*         "�����̶�":{"�о�����������":0,"��ѧ����":1,"ר��":2,"��ר":3,"����":4,"����":5,"Сѧ":6},*/
/*        "����״̬":{"δ��":0,"�ѻ�":1,"ɥż":2,"����":3},*/
/*        "ְ��":{"һ����ʽԱ��": 0,"һ�������Ա": 1,"�м�������Ա": 2,"��ǲԱ��": 3,"������": 4,"����ʽԱ��": 5,"�߼�������Ա": 6},*/
/*        "��λ����":{"����":0,"������ҵ":1,"���йɷ�":2,"������ҵ":3,"������ҵ��λ":4,"��Ӫ��ҵ":5,"�������":6,"˽Ӫ��ҵ":7}*/
/*��������*/
if  �����̶�="˶ʿ��������" then  EDUCATION=0;
if �����̶�="��ѧ����" then  EDUCATION=1;
if  �����̶�="ר��" then EDUCATION=2;
if  �����̶�="��ר" then EDUCATION=3;
if �����̶�="����" then EDUCATION=4;
if  �����̶�="����" then EDUCATION=5;
if �����̶�="Сѧ" then  EDUCATION=6;
drop �����̶�;
/*����״��*/
if ����״��="δ��" then  MARRIAGE=0;
if ����״��="�ѻ�" then MARRIAGE=1;
if ����״��="ɥż" then MARRIAGE=2;
if ����״��="����" then  MARRIAGE=3;
drop ����״��;

/*�Ա�*/
if  �Ա�="��" then GENDER1=0 ; 
if �Ա�="Ů" then GENDER1=1 ; 
drop �Ա�;

/*������Ϣ*/
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

/*��������*/
if ��������="���ػ���" then do; PERMANENT_TYPE1=0;;end;
else if ��������="����ũ��"  then do; PERMANENT_TYPE1=1;;end;
else if ��������="��ػ���" then do; PERMANENT_TYPE1=2;;end;
else PERMANENT_TYPE1=3;
drop ��������;

/*���ʷ���·��*/
/*if н�ʷ��ŷ�ʽ="�ֽ�" then SALARY_PAY_WAY1=2;*/
/*if н�ʷ��ŷ�ʽ="��" then SALARY_PAY_WAY1=1;*/
/*if н�ʷ��ŷ�ʽ="���д���" then SALARY_PAY_WAY1=3;*/
/*if н�ʷ��ŷ�ʽ="����" then SALARY_PAY_WAY1=3;*/
/*if н�ʷ��ŷ�ʽ="����" then SALARY_PAY_WAY1=0;*/
drop н�ʷ��ŷ�ʽ;
/*��������*/

/*��������*/
if ��������="�����𰴽ҹ���"  then LOCAL_RESCONDITION_G =0;
else if ��������="��˾����"  then LOCAL_RESCONDITION_G =1;
else if ��������="����ס��"  then LOCAL_RESCONDITION_G =2;
else if ��������="��ҵ���ҷ�"  then LOCAL_RESCONDITION_G =3;
else if ��������="�ް��ҹ���"  then LOCAL_RESCONDITION_G =4;
else if ��������="�Խ���"  then LOCAL_RESCONDITION_G =5;
else if ��������="����"  then LOCAL_RESCONDITION_G =6;
else if ��������="����"  then LOCAL_RESCONDITION_G =7;
drop �������� ס������ ��ְʱ��;

/*ְ��*/
if ְ��= "����ʽԱ��" then position_G =5;
else if ְ��= "������" then position_G =4;
else if ְ��= "�߼�������Ա" then position_G =6;
else if ְ��= "��ǲԱ��" then position_G =3;
else if ְ��= "һ�������Ա" then position_G =1;
else if ְ��= "һ����ʽԱ��" then position_G =0;
else if ְ��= "�м�������Ա" then position_G =2;
else position_G =7;
drop ְ��;

/*��λ����*/
if ��λ���� ="����" then comp_type=0;
else if ��λ���� ="���йɷ�" then comp_type=2;
else if ��λ���� ="������ҵ" then comp_type=1;
else if ��λ���� ="������ҵ��λ" then comp_type=4;
else if ��λ���� ="��Ӫ��ҵ" then comp_type=5;
else if ��λ���� ="�������" then comp_type=6;
else if ��λ���� ="˽Ӫ��ҵ" then comp_type=7;
else if ��λ���� ="������ҵ" then comp_type=3;
drop ��λ����;


/*��ر�ǩ*/
if  ��ر�ǩ="���"  then  nonlocal=0;
else if ��ر�ǩ="����"  then  nonlocal=1;
drop ��ر�ǩ;

/*�Ʋ���Ϣ*/
/*if �Ʋ���Ϣ = "�з��г�" then asset = 3;*/
/*else if �Ʋ���Ϣ = "�з��޳�" then asset =2;*/
/*else if �Ʋ���Ϣ = "�г��޷�" then asset = 1;*/
/*else if �Ʋ���Ϣ = "�޳��޷�" then asset = 0;*/
drop �Ʋ���Ϣ INDUSTRY_NAME;

/*����*/
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

if kindex(BRANCH_NAME,"�Ϻ�") or kindex(BRANCH_NAME,"���")or kindex(BRANCH_NAME,"����")or kindex(BRANCH_NAME,"����") then branch_=2;
else if kindex(BRANCH_NAME,"ҵ������")  then branch_=1;
else branch_=0;
drop BRANCH_NAME area;
run;



data orig.customer_info;
set customer_info;
run;

