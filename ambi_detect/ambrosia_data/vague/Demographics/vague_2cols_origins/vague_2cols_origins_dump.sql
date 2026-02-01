BEGIN TRANSACTION;
CREATE TABLE Countries (
    id INTEGER PRIMARY KEY,
    country_name TEXT
);
INSERT INTO "Countries" VALUES(1,'USA');
INSERT INTO "Countries" VALUES(2,'Canada');
INSERT INTO "Countries" VALUES(3,'Mexico');
INSERT INTO "Countries" VALUES(4,'UK');
INSERT INTO "Countries" VALUES(5,'Germany');
CREATE TABLE Education_Levels (
    education_level_id INTEGER PRIMARY KEY,
    level_description TEXT UNIQUE
);
INSERT INTO "Education_Levels" VALUES(1,'Bachelor''s Degree');
INSERT INTO "Education_Levels" VALUES(2,'Master''s Degree');
INSERT INTO "Education_Levels" VALUES(3,'PhD');
INSERT INTO "Education_Levels" VALUES(4,'High School Diploma');
INSERT INTO "Education_Levels" VALUES(5,'Associate Degree');
CREATE TABLE Employment (
    employment_id INTEGER PRIMARY KEY,
    immigrant_id INTEGER REFERENCES Immigrants(immigrant_id),
    occupation_id INTEGER REFERENCES Occupations(occupation_id),
    education_level_id INTEGER REFERENCES Education_Levels(education_level_id),
    salary REAL
);
INSERT INTO "Employment" VALUES(1,1,1,1,70000.0);
INSERT INTO "Employment" VALUES(2,2,2,2,60000.0);
INSERT INTO "Employment" VALUES(3,3,3,3,50000.0);
INSERT INTO "Employment" VALUES(4,4,4,4,80000.0);
INSERT INTO "Employment" VALUES(5,5,5,5,40000.0);
CREATE TABLE Immigrants (
    immigrant_id INTEGER PRIMARY KEY,
    country_of_origin INTEGER REFERENCES Countries(id),
    region_of_origin INTEGER REFERENCES Regions(region_id),
    age INTEGER,
    gender CHAR(1) 
);
INSERT INTO "Immigrants" VALUES(1,1,2,30,'F');
INSERT INTO "Immigrants" VALUES(2,2,3,28,'M');
INSERT INTO "Immigrants" VALUES(3,3,4,29,'F');
INSERT INTO "Immigrants" VALUES(4,4,1,32,'M');
INSERT INTO "Immigrants" VALUES(5,5,5,31,'F');
CREATE TABLE Occupations (
    occupation_id INTEGER PRIMARY KEY,
    occupation_title TEXT UNIQUE
);
INSERT INTO "Occupations" VALUES(1,'Software Engineer');
INSERT INTO "Occupations" VALUES(2,'Doctor');
INSERT INTO "Occupations" VALUES(3,'Teacher');
INSERT INTO "Occupations" VALUES(4,'Engineer');
INSERT INTO "Occupations" VALUES(5,'Architect');
CREATE TABLE Regions (
    region_id INTEGER PRIMARY KEY,
    region_name TEXT
);
INSERT INTO "Regions" VALUES(1,'Northeast');
INSERT INTO "Regions" VALUES(2,'Southwest');
INSERT INTO "Regions" VALUES(3,'Central');
INSERT INTO "Regions" VALUES(4,'East');
INSERT INTO "Regions" VALUES(5,'West');
COMMIT;
