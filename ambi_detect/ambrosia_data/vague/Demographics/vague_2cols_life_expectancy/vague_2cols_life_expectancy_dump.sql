BEGIN TRANSACTION;
CREATE TABLE EducationLevels(
    education_level_id INTEGER PRIMARY KEY,
    level_name TEXT,
    years_of_education INTEGER);
INSERT INTO "EducationLevels" VALUES(1,'High School Diploma',12);
INSERT INTO "EducationLevels" VALUES(2,'Bachelor''s Degree',16);
CREATE TABLE LifeExpectancies(
    id INTEGER PRIMARY KEY,
    average_years_of_life REAL,
    gender_specific_life_expectancy REAL,
    region_id INTEGER,
    FOREIGN KEY(region_id) REFERENCES Regions(region_id));
INSERT INTO "LifeExpectancies" VALUES(1,75.5,73.2,1);
INSERT INTO "LifeExpectancies" VALUES(2,78.1,75.8,2);
CREATE TABLE Occupations(
    occupation_id INTEGER PRIMARY KEY,
    job_title TEXT,
    industry TEXT);
INSERT INTO "Occupations" VALUES(1,'Software Engineer','Technology');
INSERT INTO "Occupations" VALUES(2,'Teacher','Education');
CREATE TABLE PersonEducation(
    person_education_id INTEGER PRIMARY KEY,
    persons_id INTEGER,
    education_levels_id INTEGER,
    FOREIGN KEY(persons_id) REFERENCES Persons(person_id),
    FOREIGN KEY(education_levels_id) REFERENCES EducationLevels(education_level_id));
CREATE TABLE PersonOccupation(
    person_occupation_id INTEGER PRIMARY KEY,
    persons_id INTEGER,
    occupations_id INTEGER,
    FOREIGN KEY(persons_id) REFERENCES Persons(person_id),
    FOREIGN KEY(occupations_id) REFERENCES Occupations(occupation_id));
CREATE TABLE Persons(
    person_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    birthdate DATE);
INSERT INTO "Persons" VALUES(1,'John','Doe','1980-06-23');
INSERT INTO "Persons" VALUES(2,'Jane','Smith','1978-09-15');
CREATE TABLE Regions(
    region_id INTEGER PRIMARY KEY,
    name TEXT,
    population INTEGER);
INSERT INTO "Regions" VALUES(1,'Region A',5000000);
INSERT INTO "Regions" VALUES(2,'Region B',3000000);
COMMIT;
