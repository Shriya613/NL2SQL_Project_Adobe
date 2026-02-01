BEGIN TRANSACTION;
CREATE TABLE Classes(
    classId INTEGER PRIMARY KEY,
    className TEXT,
    startDate DATE,
    endDate DATE,
    teacherId INTEGER,
    FOREIGN KEY (teacherId) REFERENCES Teachers(teacherId)
);
INSERT INTO "Classes" VALUES(1,'Math101','2023-09-01','2023-12-21',1);
INSERT INTO "Classes" VALUES(2,'Physics101','2023-08-30','2023-12-20',2);
INSERT INTO "Classes" VALUES(3,'Chemistry101','2023-09-01','2023-12-21',3);
INSERT INTO "Classes" VALUES(4,'Biology101','2023-08-30','2023-12-20',4);
INSERT INTO "Classes" VALUES(5,'History101','2023-09-01','2023-12-21',5);
CREATE TABLE Difficulty_levels(
    difficultyLevel TEXT PRIMARY KEY,
    levelDescription TEXT
);
INSERT INTO "Difficulty_levels" VALUES('Beginner','For beginners with little or no prior knowledge.');
INSERT INTO "Difficulty_levels" VALUES('Intermediate','For those who have a good understanding of the basics.');
INSERT INTO "Difficulty_levels" VALUES('Advanced','For experienced learners looking for challenging content.');
CREATE TABLE Learning_Materials(
    materialId INTEGER PRIMARY KEY,
    typeOfMaterial TEXT ,
    title TEXT,
    description TEXT,
    difficultyLevel TEXT,
    FOREIGN KEY (difficultyLevel) REFERENCES Difficulty_levels(difficultyLevel)
);
INSERT INTO "Learning_Materials" VALUES(1,'Textbook','Introduction to Mathematics','A comprehensive textbook covering basic mathematics concepts.','Intermediate');
INSERT INTO "Learning_Materials" VALUES(2,'Textbook','General Chemistry','An introductory chemistry textbook suitable for college students.','Beginner');
INSERT INTO "Learning_Materials" VALUES(3,'Online Course','Advanced Physics','A detailed online course covering advanced physics topics.','Advanced');
INSERT INTO "Learning_Materials" VALUES(4,'Online Course','World History I','An interactive online course exploring world history up to 1500 AD.','Intermediate');
CREATE TABLE Students(
    studentId INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    enrollmentYear INTEGER,
    classId INTEGER,
    FOREIGN KEY (classId) REFERENCES Classes(classId)
);
INSERT INTO "Students" VALUES(1,'Alice','Ward',2023,1);
INSERT INTO "Students" VALUES(2,'Bob','Green',2023,2);
INSERT INTO "Students" VALUES(3,'Charlie','Black',2023,3);
INSERT INTO "Students" VALUES(4,'David','White',2023,4);
INSERT INTO "Students" VALUES(5,'Eva','Gray',2023,5);
CREATE TABLE Subjects(
    id INTEGER PRIMARY KEY,
    subjectName TEXT
);
INSERT INTO "Subjects" VALUES(1,'Mathematics');
INSERT INTO "Subjects" VALUES(2,'Physics');
INSERT INTO "Subjects" VALUES(3,'Chemistry');
INSERT INTO "Subjects" VALUES(4,'Biology');
INSERT INTO "Subjects" VALUES(5,'History');
CREATE TABLE Teachers(
    teacherId INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Teachers" VALUES(1,'John','Doe','john.doe@example.com');
INSERT INTO "Teachers" VALUES(2,'Jane','Smith','jane.smith@example.com');
INSERT INTO "Teachers" VALUES(3,'Michael','Johnson','michael.johnson@example.com');
INSERT INTO "Teachers" VALUES(4,'Emily','Williams','emily.williams@example.com');
INSERT INTO "Teachers" VALUES(5,'David','Brown','david.brown@example.com');
COMMIT;
