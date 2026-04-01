


#These data were obtained from the Current Population Survey at https://www.census.gov/programs-surveys/cps.html
#
#To use these data you should add a reference to:
#
#Flood, Sarah and King, Miriam and Rodgers, Renae and Ruggles, Steven and Warren, J Robert (2020).
#Integrated Public Use Microdata Series, Current Population Survey: Version 7.0 [dataset]
#Minneapolis, MN: IPUMS, 10.18128/D030.V7.0
#
# See https://cps.ipums.org/cps/citation.shtml for the rules regarding citation and use of the CPS database

library(tidyverse)
library(here)
source("https://www.ics.uci.edu/~akseong/Rcode/EDA_R_functions.R")

load(here::here("data", "salary.RData"))
table(table(salary$householdid))
table(salary$female)
table(salary$hispanic)
table(salary$marital)
table(salary$employment)
table(salary$edu)
table(salary$age)
hist(salary$age)
table(salary$race)
table(salary$citizen)
table(salary$nativity)
table(salary$labforce)
table(salary$occ)
table(salary$wkstat)
table(salary$schlcoll)
table(salary$firmsize)
table(salary$difficulty)
table(salary$classworker)
table(salary$movedstate)
hist(salary$famincome)
hist(salary$personalincome)
hist(salary$incomewage)
table(salary$hoursworked)

# salary data:  38,208 obs of 23 vars, no missingness
#   - year: all 2019 (38208); 
#   - householdid: mostly 1 (20079) or 2 (7271); 3 (878); 4 (165); 5 (41); 6 (10); 7 (4)
#   - female: binary, 1 (18050 ), 0 (20158)
#   - hispanic: binary, 1 (9046), 0 (29162)
#   - marital: Married (21712), NeverMarried (11372), Divorced (3827), Separated (826), Widowed (471)
#   - employment: At work (38208), all others 0 (armed forces, has job, not at work in last wek, not in labor force, unemployed)
#   - edu: CG (10977), HSD (12150), No HSD (3342), SC (11739)
#   - age: 18-65
#   - race: Asian (2521), Black (4918), Hawaiian/Pacific Islander (262), Native American (605), White (29902)
#   - citizen: Born US (30282), Not citizen (4071), Citizen (3543), Born abroad, US parents (312)
#   - nativity: 0 (53), 1 (26389), 2 (609), 3 (609), 4 (2358), 5 (8190)
#   - labforce: 2 (38208)
#   - occ: primary occupational sector:
#      - architect/engineer 488
#      - arts/sports/media 423
#      - business operations 859
#      - computer/maths 1110
#      - construction 2102
#      - education 565
#      - extraction 69
#      - farming 260
#      - finance 735
#      - food 1578
#      - health 2767
#      - installation 1355
#      - legal 159
#      - maintenance 1228
#      - management 2747
#      - office 4298
#      - personal care 839
#      - production 2789
#      - protective 276  - protective services, e.g. law enforcement, bodyguard, lifeguard, security officers
#      - sales 2906
#      - science 159
#      - social service 344
#      - technician 130
#      - transportation 2152
#   - wkstat: full-time (38208), Part-time, Unemployed
#   - schlcoll: Does not attend school (30028), College or University (1216), High school (31), NIU (6903)
#   - firmsize: 0 (659), 1 (1-24; 11320), 2 (25-99; 2898), 3 (100-499; 5075), 4 (500-999; 2099). 5 (>1000; 16157)
#   - difficulty: 0 (37139), 1 (1069)
#   - classworker: NA (659), Government employee (5198), Self-employed (2006), Unpaid family worked (7), Wage/salary (30338)
#   - movedstate: No (34694), From US (3416), From out US (98)
#   - famincome
#   - personalincome
#   - incomewage
#   - hoursworked: 35-40

# dataset already subsetted to:
#   - labforce = 2    ?
#   - employment = "At work"
#   - age btwn 18-65
#   - year 2019
#   - occ: no "military", no "unemployed/never worked"
#   - wkstat = "full-time" (no part-time, unemployed)

# level clarifications:
# nativity: 0 unknown, 1 Native-born, both parents native; 2 native-born, father foreign, mother native; 3 native-born, mother foreign, father native; 4 native-born, both parents foreign; 5 foreign-born
# schlcoll: 0 = NIU ("not in universe" or not applicable), 1 = HS fulltime, 2 = HS part-time, 3 = Coll/Univ fulltime, 4 = Coll/Univ parttime, 5 does not attend
# difficulty: binary; any physical or cognitive difficulty (diffhear, diffeye, diffrem, diffphys, diffmob, diffcare (personal care))


# also should subset by classworker == wage/salary,
# difficulty = 0  (no reported physical/cognitive difficulty)

sal <- salary %>% 
  filter(
    age >= 18,
    age <= 65,
    employment == "At work",
    !(occ %in% c("unemployed/never worked","military")),
    hoursworked >= 35,
    hoursworked <= 40,
    wkstat == "Full-time",
    classworker == "Wage/salary"
  ) %>% 
  dplyr::select(
    incomewage, female, hispanic, marital, edu, age, race, citizen, nativity, occ, schlcoll, firmsize, difficulty, hoursworked
  )
levels(salary$classworker)
table(sal$citizen, sal$nativity)
table(sal$nativity, sal$citizen)
length(table(sal$occ))
table(sal$schlcoll)
# of interest: female (binary), hispanic (binary), marital (5 levels), edu (4 levels), age, race (5 levels), occ (26 levels, 2 empty - military, unemployed) 
# unsure: citizen (4 levels), nativity (5 levels), schlcoll (4 levels), firmsize (5 levels, 1 empty - size 0), difficulty, hoursworked?
# - nativity 0 = unknown; levels 2-3 should be combined (1 parent foreign)
# - citizen and/or nativity?
# - schlcoll probably not relevant / subsumed by edu
# - firmsize: confounder, even across female, uneven across occupation, probably can reduce to 4 levels (just >500; 500-999 is the smallest category anyway; not sure a qualitative difference between 500 and >1000 categories)
table(sal$female, sal$firmsize)
table(sal$firmsize, sal$occ)
# - difficulty: confounder, but small and roughly similar proportions (0.0273 males, 0.0235 females).  Filter out?
table(sal$female, sal$difficulty)



table(sal$female, sal$occ)
# - reduce occ levels: 
#    - Math/science: architect/engineer 488 (7M), computer/maths 1110 (3M), science 159
#    - arts/sports/media 423
#    - business operations 859 (1.5F), finance 735 (1.75F), legal 159 (8F), management 2747, office 4298 (3F), sales 2906
#    - construction 2102 (30M), extraction 69 (16M)
#    - installation 1355 (30M),  maintenance 1228, technician 130 (4M), production 2789 (2M)
#    - education 565 (4F)
#    - farming 260 (2M)
#    - food 1578 
#    - health 2767 (6F), personal care 839 (3F)
#    - protective 276 (3M)
#    - social service 344 (2.5F)
#    - transportation 2152 (4.5 M)

# could reduce to roughly equal / male/female-dominated industries:
#    - roughly equal: science, arts/sports/media, sales, maintenance, food
#    - F dominated: business operations, finance, legal, office, education, health, personal care, social service
#    - M dominated: architect/engineer, computer/maths, construction, extraction, installation, technician, production, farming, protective, transportation

# could reduce to roughly equal / male/female-dominated industries and skilled (skilled meaning sector generally requires college education or more):
#    - roughly equal, skilled: science, 
#    - roughly equal, unskilled: arts/sports/media, sales, maintenance, food
#    - F dominated, skilled: business operations, finance, legal, education, health
#    - F dominated, unskilled: office, personal care, social service
#    - M dominated, skilled: architect/engineer, computer/maths, 
#    - M dominated, unskilled: construction, extraction, installation, technician, production, farming, protective, transportation

















#SELECT CASES. AGED 18-65, SINGLE RACE, NON-MILITARY EMPLOYED 35-40H/WEEK
data= filter(salary, incomewage>=500, age>=18, age<=65, employment=="At work", !(occ %in% c("unemployed/neverworked","military")), hoursworked>=35, hoursworked<=40, wkstat=="Full-time") |>
  mutate(occ= factor(occ), logincome= log10(incomewage), black= ifelse(race=='Black',1,0), college= ifelse(edu=='CG',1,0), classworker=factor(classworker)) |>
  mutate(female_black= female * black) |>
  select(c(logincome, age, female, hispanic, black, college, occ, classworker, female_black))


y= data$logincome
z= data$age
x.adjust= model.matrix(~ occ, data=data)
x= data[,c('female','hispanic','black','college','classworker')]
x= mutate(x, government= as.numeric(classworker=='Government employee'), selfemployed= as.numeric(classworker=='Self-employed')) |>
  select(-classworker)

