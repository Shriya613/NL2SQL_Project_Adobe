BEGIN TRANSACTION;
CREATE TABLE Course(
    courseId INTEGER PRIMARY KEY,
    title TEXT,
    description TEXT,
    duration INT ,
    educatorId INTEGER,
    FOREIGN KEY (educatorId) REFERENCES Educator(id)
);
INSERT INTO "Course" VALUES(1,'Introduction to Calculus','A basic introduction to calculus.',120,1);
INSERT INTO "Course" VALUES(2,'General Physics I','An overview of fundamental physics concepts.',150,2);
INSERT INTO "Course" VALUES(3,'Advanced Biology','Exploring complex biological systems.',180,3);
INSERT INTO "Course" VALUES(4,'Data Structures and Algorithms','Learn about various data structures and algorithms.',150,4);
INSERT INTO "Course" VALUES(5,'Great American Novels','Discussion on major works by classic American authors.',120,5);
CREATE TABLE Educator(
    id INTEGER PRIMARY KEY,
    name TEXT,
    specialization TEXT,
    yearsOfExperience INT 
);
INSERT INTO "Educator" VALUES(1,'John Doe','Mathematics',8);
INSERT INTO "Educator" VALUES(2,'Jane Smith','Physics',7);
INSERT INTO "Educator" VALUES(3,'Alice Brown','Biology',6);
INSERT INTO "Educator" VALUES(4,'Bob Johnson','Computer Science',9);
INSERT INTO "Educator" VALUES(5,'Charlie White','English Literature',3);
CREATE TABLE Student(
    studentId INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    dateOfBirth DATE,
    enrolledCourses TEXT[] -- Assuming we can use array type for multiple courses here
);
INSERT INTO "Student" VALUES(1,'Tom','Hanks','1990-05-22',NULL);
INSERT INTO "Student" VALUES(2,'Emma','Watson','1990-11-15',NULL);
INSERT INTO "Student" VALUES(3,'Leonardo','DiCaprio','1974-11-11',NULL);
INSERT INTO "Student" VALUES(4,'Kate','Winslet','1975-10-05',NULL);
INSERT INTO "Student" VALUES(5,'Robert','Downey Jr','1965-04-04',NULL);
CREATE TABLE Venue(
    venueId INTEGER PRIMARY KEY,
    name TEXT,
    location TEXT,
    educationalVenue TEXT,
    ticketPrice TEXT
);
INSERT INTO "Venue" VALUES(1,'The Museum','Central City','Museum','$15');
INSERT INTO "Venue" VALUES(2,'Stellar Planetarium','South Side','Planetarium','$16');
INSERT INTO "Venue" VALUES(3,'Natural History Museum','West End','Museum','$17');
INSERT INTO "Venue" VALUES(4,'Galaxy Planetarium','East Town','Planetarium','$15');
INSERT INTO "Venue" VALUES(5,'Art Gallery','North Point','Art Exhibition Hall','Free Entry');
COMMIT;
