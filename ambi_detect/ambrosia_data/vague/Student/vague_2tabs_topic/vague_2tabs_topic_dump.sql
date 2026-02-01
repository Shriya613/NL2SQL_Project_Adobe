BEGIN TRANSACTION;
CREATE TABLE Industry_Sector (
    id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT
);
INSERT INTO "Industry_Sector" VALUES(1,'Technology','Includes companies involved in software development, hardware manufacturing and IT services.');
INSERT INTO "Industry_Sector" VALUES(2,'Finance','Companies dealing with financial transactions, investments and banking operations.');
INSERT INTO "Industry_Sector" VALUES(3,'Healthcare','Organizations providing medical care, research and pharmaceutical products.');
INSERT INTO "Industry_Sector" VALUES(4,'Education','Institutions focused on teaching, training and learning activities.');
INSERT INTO "Industry_Sector" VALUES(5,'Retail','Businesses that sell goods and services directly to consumers.');
CREATE TABLE Internship (
    id INTEGER PRIMARY KEY,
    student_id INTEGER,
    industry_sector_id INTEGER,
    project_id INTEGER,
    FOREIGN KEY(student_id) REFERENCES Students(id),
    FOREIGN KEY(industry_sector_id) REFERENCES Industry_Sector(id),
    FOREIGN KEY(project_id) REFERENCES Project_Name(id)
);
INSERT INTO "Internship" VALUES(1,1,1,1);
INSERT INTO "Internship" VALUES(2,2,2,2);
INSERT INTO "Internship" VALUES(3,3,3,3);
INSERT INTO "Internship" VALUES(4,4,4,4);
INSERT INTO "Internship" VALUES(5,5,5,5);
CREATE TABLE Project_Name (
    id INTEGER PRIMARY KEY,
    project_name TEXT,
    start_date DATE,
    end_date DATE
);
INSERT INTO "Project_Name" VALUES(1,'Project Alpha','2021-08-01','2021-09-30');
INSERT INTO "Project_Name" VALUES(2,'Project Beta','2021-10-01','2021-11-30');
INSERT INTO "Project_Name" VALUES(3,'Project Gamma','2021-12-01','2022-01-31');
INSERT INTO "Project_Name" VALUES(4,'Project Delta','2022-02-01','2022-03-31');
INSERT INTO "Project_Name" VALUES(5,'Project Epsilon','2022-04-01','2022-05-31');
COMMIT;
