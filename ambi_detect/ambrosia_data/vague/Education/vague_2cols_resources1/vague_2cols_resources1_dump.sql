BEGIN TRANSACTION;
CREATE TABLE Assignments (
    ID INTEGER PRIMARY KEY,
    Title TEXT,
    Description TEXT,
    DueDate DATE,
    MaxPoints INTEGER
);
INSERT INTO "Assignments" VALUES(1,'Vocabulary Quiz','Test your knowledge of vocabulary words learned this week','2023-03-15',50);
INSERT INTO "Assignments" VALUES(2,'Research Paper','Write a research paper on a topic related to our current textbook','2023-04-10',100);
INSERT INTO "Assignments" VALUES(3,'Creative Writing','Write a short story based on a scene from a textbook','2023-03-25',80);
INSERT INTO "Assignments" VALUES(4,'Group Presentation','Present a group project on a topic related to the unit','2023-04-15',100);
INSERT INTO "Assignments" VALUES(5,'Final Exam','Review all material covered during the unit','2023-04-20',100);
CREATE TABLE Classrooms (
    ID INTEGER PRIMARY KEY,
    RoomNumber INTEGER,
    Building TEXT,
    Capacity INTEGER
);
INSERT INTO "Classrooms" VALUES(1,101,'Main',30);
INSERT INTO "Classrooms" VALUES(2,201,'Annex',25);
INSERT INTO "Classrooms" VALUES(3,301,'Main',35);
INSERT INTO "Classrooms" VALUES(4,401,'Annex',20);
INSERT INTO "Classrooms" VALUES(5,501,'Library',60);
CREATE TABLE Grades (
    ID INTEGER PRIMARY KEY,
    StudentID INTEGER,
    AssignmentID INTEGER,
    Score INTEGER,
    Feedback TEXT,
    FOREIGN KEY (StudentID) REFERENCES Students(ID),
    FOREIGN KEY (AssignmentID) REFERENCES Assignments(ID)
);
INSERT INTO "Grades" VALUES(1,1,1,45,'Good effort but needs improvement');
INSERT INTO "Grades" VALUES(2,2,1,60,'Great job! Keep up the good work');
INSERT INTO "Grades" VALUES(3,3,1,75,'Excellent work! Very well done');
INSERT INTO "Grades" VALUES(4,4,1,85,'Outstanding performance! Well done');
INSERT INTO "Grades" VALUES(5,5,1,90,'Exceptional work! You excelled at this assignment');
CREATE TABLE Students (
    ID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    GradeLevel INTEGER
);
INSERT INTO "Students" VALUES(1,'John','Doe',10);
INSERT INTO "Students" VALUES(2,'Jane','Smith',9);
INSERT INTO "Students" VALUES(3,'Emily','Brown',8);
INSERT INTO "Students" VALUES(4,'Michael','Johnson',12);
INSERT INTO "Students" VALUES(5,'Sarah','White',7);
CREATE TABLE Teachers (
    ID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    SubjectTaught TEXT
);
INSERT INTO "Teachers" VALUES(1,'Mr.','Smith','Math');
INSERT INTO "Teachers" VALUES(2,'Ms.','Williams','English');
INSERT INTO "Teachers" VALUES(3,'Dr.','Harris','History');
INSERT INTO "Teachers" VALUES(4,'Mrs.','Davis','Science');
INSERT INTO "Teachers" VALUES(5,'Mr.','Martinez','Art');
CREATE TABLE Textbooks_Supplies_Projects (
    ID INTEGER PRIMARY KEY,
    TextbookTitle TEXT,
    SupplyItem TEXT,
    ProjectDescription TEXT);
INSERT INTO "Textbooks_Supplies_Projects" VALUES(1,'To Kill a Mockingbird','Poster paper','Create a poster about the novel');
INSERT INTO "Textbooks_Supplies_Projects" VALUES(2,'The Great Gatsby','Pencils','Write a short story inspired by the book');
INSERT INTO "Textbooks_Supplies_Projects" VALUES(3,'A Wrinkle in Time','Markers','Draw a map from the book');
INSERT INTO "Textbooks_Supplies_Projects" VALUES(4,'Romeo and Juliet','Glue sticks','Make a collage about the play');
INSERT INTO "Textbooks_Supplies_Projects" VALUES(5,'Of Mice and Men','Index cards','Create flashcards to memorize quotes');
COMMIT;
