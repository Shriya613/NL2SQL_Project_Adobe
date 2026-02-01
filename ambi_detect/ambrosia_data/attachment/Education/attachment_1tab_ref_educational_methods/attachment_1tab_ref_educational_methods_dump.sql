BEGIN TRANSACTION;
CREATE TABLE Classes(
    classID INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    startDate DATE,
    endDate DATE,
    EducatorID INT REFERENCES Educators(educatorID),
    FOREIGN KEY (classID) REFERENCES Enrollments(classID));
INSERT INTO "Classes" VALUES(1,'Math Class A','2023-09-01','2023-12-20',1);
INSERT INTO "Classes" VALUES(2,'English Class B','2023-09-01','2023-12-20',2);
CREATE TABLE CourseMaterials(
    materialID INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    url TEXT UNIQUE);
INSERT INTO "CourseMaterials" VALUES(1,'Algebra Book','https://example.com/algebra-book');
INSERT INTO "CourseMaterials" VALUES(2,'Grammar Guide','https://example.com/grammar-guide');
CREATE TABLE Curriculums(
    curriculumID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    EducatorID INT REFERENCES Educators(educatorID));
INSERT INTO "Curriculums" VALUES(1,'Grade 6 Mathematics',1);
INSERT INTO "Curriculums" VALUES(2,'Grade 7 English',2);
CREATE TABLE Educators(
    educatorID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    subject TEXT);
INSERT INTO "Educators" VALUES(1,'Michael','Johnson','Mathematics');
INSERT INTO "Educators" VALUES(2,'Emily','Williams','English');
CREATE TABLE Enrollments(
    enrollmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INT REFERENCES Students(studentID),
    classID INT REFERENCES Classes(classID),
    courseMaterialID INT REFERENCES CourseMaterials(materialID));
INSERT INTO "Enrollments" VALUES(1,1,1,1);
INSERT INTO "Enrollments" VALUES(2,2,2,2);
CREATE TABLE Lessons(
    lessonID INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT,
    educationalMethod TEXT,
    curriculumID INT REFERENCES Curriculums(curriculumID),
    classID INT REFERENCES Classes(classID));
INSERT INTO "Lessons" VALUES(1,'Lesson on linear equations','Online Learning',1,1);
INSERT INTO "Lessons" VALUES(2,'Lesson on sentence structure','Online Learning',2,2);
INSERT INTO "Lessons" VALUES(3,'Introduction to algebra','Classroom Teaching',1,3);
INSERT INTO "Lessons" VALUES(4,'Vocabulary building','Classroom Teaching',2,4);
CREATE TABLE Students(
    studentID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    dateOfBirth DATE);
INSERT INTO "Students" VALUES(1,'John','Doe','2003-01-01');
INSERT INTO "Students" VALUES(2,'Jane','Smith','2004-02-02');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',2);
INSERT INTO "sqlite_sequence" VALUES('Educators',2);
INSERT INTO "sqlite_sequence" VALUES('Classes',2);
INSERT INTO "sqlite_sequence" VALUES('Curriculums',2);
INSERT INTO "sqlite_sequence" VALUES('CourseMaterials',2);
INSERT INTO "sqlite_sequence" VALUES('Enrollments',2);
INSERT INTO "sqlite_sequence" VALUES('Lessons',4);
COMMIT;
