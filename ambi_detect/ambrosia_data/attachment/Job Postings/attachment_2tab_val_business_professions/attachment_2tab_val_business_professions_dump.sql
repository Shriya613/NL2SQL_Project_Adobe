BEGIN TRANSACTION;
CREATE TABLE Data_Analyst(
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INT,
    experience INT,
    software_used TEXT
);
INSERT INTO "Data_Analyst" VALUES(1,'Mary Taylor',29,7,'Excel');
INSERT INTO "Data_Analyst" VALUES(2,'James Jackson',30,8,'Tableau');
INSERT INTO "Data_Analyst" VALUES(3,'Patrick Adams',31,5,'PowerBI');
INSERT INTO "Data_Analyst" VALUES(4,'Charles White',28,4,'SQL');
INSERT INTO "Data_Analyst" VALUES(5,'Eric Harris',33,6,'SPSS');
CREATE TABLE HR_Recruiter(
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INT,
    experience INT,
    recruitment_strategy TEXT
);
INSERT INTO "HR_Recruiter" VALUES(1,'Linda Martinez',37,10,'Behavioral Interviews');
INSERT INTO "HR_Recruiter" VALUES(2,'Michelle Green',29,6,'Job Analysis');
INSERT INTO "HR_Recruiter" VALUES(3,'Barbara Nelson',30,4,'Competency Modeling');
INSERT INTO "HR_Recruiter" VALUES(4,'Karen Young',31,7,'Targeted Selection');
INSERT INTO "HR_Recruiter" VALUES(5,'Ann Turner',32,5,'Assessment Centers');
CREATE TABLE Marketing_Specialist(
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INT,
    experience INT,
    working_arrangement TEXT 
);
INSERT INTO "Marketing_Specialist" VALUES(1,'Michael Jones',35,7,'Hybrid');
INSERT INTO "Marketing_Specialist" VALUES(2,'Laura Wilson',30,5,'Remote');
INSERT INTO "Marketing_Specialist" VALUES(3,'David Miller',28,3,'Onsite');
INSERT INTO "Marketing_Specialist" VALUES(4,'Helen Moore',33,8,'Hybrid');
INSERT INTO "Marketing_Specialist" VALUES(5,'Paul Thompson',27,4,'Onsite');
CREATE TABLE Sales_Manager(
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INT,
    experience INT,
    working_arrangement TEXT 
);
INSERT INTO "Sales_Manager" VALUES(1,'John Doe',32,8,'Hybrid');
INSERT INTO "Sales_Manager" VALUES(2,'Jane Smith',40,6,'Remote');
INSERT INTO "Sales_Manager" VALUES(3,'Alice Johnson',29,4,'Onsite');
INSERT INTO "Sales_Manager" VALUES(4,'Bob Brown',37,10,'Hybrid');
INSERT INTO "Sales_Manager" VALUES(5,'Emily Davis',26,3,'Onsite');
CREATE TABLE Software_Developer(
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INT,
    experience INT,
    programming_language TEXT
);
INSERT INTO "Software_Developer" VALUES(1,'Robert Martin',31,10,'Java');
INSERT INTO "Software_Developer" VALUES(2,'William Anderson',36,6,'Python');
INSERT INTO "Software_Developer" VALUES(3,'Thomas Clark',29,4,'C++');
INSERT INTO "Software_Developer" VALUES(4,'Daniel Lewis',27,8,'JavaScript');
INSERT INTO "Software_Developer" VALUES(5,'Steven King',32,5,'C#');
COMMIT;
