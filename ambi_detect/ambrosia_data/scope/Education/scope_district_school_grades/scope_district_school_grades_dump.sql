BEGIN TRANSACTION;
CREATE TABLE District (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "District" VALUES(1,'District A','New York');
INSERT INTO "District" VALUES(2,'District B','California');
INSERT INTO "District" VALUES(3,'District C','Texas');
INSERT INTO "District" VALUES(4,'District D','Florida');
INSERT INTO "District" VALUES(5,'District E','Illinois');
CREATE TABLE District_School_Grades (
    district_id INTEGER,
    school_grade_id INTEGER,
    FOREIGN KEY (district_id) REFERENCES District (id),
    FOREIGN KEY (school_grade_id) REFERENCES School_Grades (id));
INSERT INTO "District_School_Grades" VALUES(1,1);
INSERT INTO "District_School_Grades" VALUES(1,2);
INSERT INTO "District_School_Grades" VALUES(2,1);
INSERT INTO "District_School_Grades" VALUES(2,2);
INSERT INTO "District_School_Grades" VALUES(3,1);
INSERT INTO "District_School_Grades" VALUES(3,2);
INSERT INTO "District_School_Grades" VALUES(4,1);
INSERT INTO "District_School_Grades" VALUES(4,2);
INSERT INTO "District_School_Grades" VALUES(5,1);
INSERT INTO "District_School_Grades" VALUES(5,2);
CREATE TABLE School_Grades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grade VARCHAR(20),
    description TEXT);
INSERT INTO "School_Grades" VALUES(1,'Kindergarten','Early childhood education program');
INSERT INTO "School_Grades" VALUES(2,'1st Grade','First year of primary education');
INSERT INTO "School_Grades" VALUES(3,'2nd Grade','Second year of primary education');
INSERT INTO "School_Grades" VALUES(4,'3rd Grade','Third year of primary education');
INSERT INTO "School_Grades" VALUES(5,'4th Grade','Fourth year of primary education');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('District',5);
INSERT INTO "sqlite_sequence" VALUES('School_Grades',5);
COMMIT;
