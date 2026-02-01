BEGIN TRANSACTION;
CREATE TABLE Course_Instructor(
    courseID INTEGER,
    instructorID INTEGER,
    startDate DATE,
    endDate DATE,
    FOREIGN KEY(courseID) REFERENCES Courses(courseID),
    FOREIGN KEY(instructorID) REFERENCES Instructors(instructorID));
INSERT INTO "Course_Instructor" VALUES(1,1,'2021-01-01','2022-12-31');
INSERT INTO "Course_Instructor" VALUES(2,2,'2021-02-01','2022-12-31');
INSERT INTO "Course_Instructor" VALUES(3,3,'2021-03-01','2022-12-31');
INSERT INTO "Course_Instructor" VALUES(4,4,'2021-04-01','2022-12-31');
INSERT INTO "Course_Instructor" VALUES(5,5,'2021-05-01','2022-12-31');
CREATE TABLE Courses(
    courseID INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT);
INSERT INTO "Courses" VALUES(1,'English_As_A_Second_Language','This course will teach you English as a second language.');
INSERT INTO "Courses" VALUES(2,'Programming_Basics','This course covers the basics of programming languages.');
INSERT INTO "Courses" VALUES(3,'Mathematics_For_Engineers','This course is designed for engineering students who need a strong foundation in mathematics.');
INSERT INTO "Courses" VALUES(4,'Artificial_Intelligence','This advanced course covers artificial intelligence concepts and applications.');
INSERT INTO "Courses" VALUES(5,'Data_Science','This comprehensive course teaches data science skills and techniques.');
CREATE TABLE Enrolled_Students(
    studentID INTEGER,
    courseID INTEGER,
    enrollmentDate DATE,
    FOREIGN KEY(studentID) REFERENCES Users(userID),
    FOREIGN KEY(courseID) REFERENCES Courses(courseID));
INSERT INTO "Enrolled_Students" VALUES(1,1,'2021-07-01');
INSERT INTO "Enrolled_Students" VALUES(2,2,'2021-06-20');
INSERT INTO "Enrolled_Students" VALUES(3,3,'2021-08-15');
INSERT INTO "Enrolled_Students" VALUES(4,4,'2021-09-01');
INSERT INTO "Enrolled_Students" VALUES(5,5,'2021-10-01');
CREATE TABLE Instructors(
    instructorID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    email TEXT);
INSERT INTO "Instructors" VALUES(1,'John','Doe','johnd@example.com');
INSERT INTO "Instructors" VALUES(2,'Jane','Smith','janes@example.com');
INSERT INTO "Instructors" VALUES(3,'Bob','Brown','bobb@example.com');
INSERT INTO "Instructors" VALUES(4,'Alice','White','alicw@example.com');
INSERT INTO "Instructors" VALUES(5,'Emily','Black','emilbl@example.com');
CREATE TABLE Online_Learning_Platforms(
    platformID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    website TEXT);
INSERT INTO "Online_Learning_Platforms" VALUES(1,'Udemy','www.udemy.com');
INSERT INTO "Online_Learning_Platforms" VALUES(2,'Coursera','www.coursera.org');
INSERT INTO "Online_Learning_Platforms" VALUES(3,'edX','www.edx.org');
INSERT INTO "Online_Learning_Platforms" VALUES(4,'Codecademy','www.codecademy.com');
INSERT INTO "Online_Learning_Platforms" VALUES(5,'LinkedIn Learning','learning.linkedin.com');
CREATE TABLE Online_Learning_Platforms_Courses(
    platformID INTEGER,
    courseID INTEGER,
    availabilityStatus TEXT ,
    FOREIGN KEY(platformID) REFERENCES Online_Learning_Platforms(platformID),
    FOREIGN KEY(courseID) REFERENCES Courses(courseID));
INSERT INTO "Online_Learning_Platforms_Courses" VALUES(1,1,'Available');
INSERT INTO "Online_Learning_Platforms_Courses" VALUES(2,1,'Unavailable');
INSERT INTO "Online_Learning_Platforms_Courses" VALUES(3,2,'Available');
INSERT INTO "Online_Learning_Platforms_Courses" VALUES(4,3,'Unavailable');
INSERT INTO "Online_Learning_Platforms_Courses" VALUES(5,4,'Available');
INSERT INTO "Online_Learning_Platforms_Courses" VALUES(3,1,'Available');
INSERT INTO "Online_Learning_Platforms_Courses" VALUES(4,1,'Unavailable');
INSERT INTO "Online_Learning_Platforms_Courses" VALUES(5,1,'Available');
CREATE TABLE Ratings(
    ratingID INTEGER PRIMARY KEY AUTOINCREMENT,
    studentID INTEGER,
    courseID INTEGER,
    stars INT ,
    reviewText TEXT,
    submissionDate DATE,
    FOREIGN KEY(studentID) REFERENCES Users(userID),
    FOREIGN KEY(courseID) REFERENCES Courses(courseID));
INSERT INTO "Ratings" VALUES(1,1,1,5,'Excellent course! Highly recommended.','2021-08-10');
INSERT INTO "Ratings" VALUES(2,2,2,4,'Good course but could use more examples.','2021-07-15');
INSERT INTO "Ratings" VALUES(3,3,3,5,'Very helpful for my studies.','2021-08-20');
INSERT INTO "Ratings" VALUES(4,4,4,5,'Great course on AI, highly informative.','2021-09-05');
INSERT INTO "Ratings" VALUES(5,5,5,4,'The content was good, but I felt it lacked depth at times.','2021-10-10');
CREATE TABLE Users(
    userID INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    password TEXT);
INSERT INTO "Users" VALUES(1,'john_doe','password123');
INSERT INTO "Users" VALUES(2,'jane_smith','securepass');
INSERT INTO "Users" VALUES(3,'bob_brown','mypassword');
INSERT INTO "Users" VALUES(4,'alice_white','strongPass#1');
INSERT INTO "Users" VALUES(5,'emily_black','mysecretpass');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Users',5);
INSERT INTO "sqlite_sequence" VALUES('Online_Learning_Platforms',5);
INSERT INTO "sqlite_sequence" VALUES('Courses',5);
INSERT INTO "sqlite_sequence" VALUES('Ratings',5);
INSERT INTO "sqlite_sequence" VALUES('Instructors',5);
COMMIT;
