BEGIN TRANSACTION;
CREATE TABLE Classes (
    classId INTEGER PRIMARY KEY AUTOINCREMENT,
    className TEXT,
    professorName TEXT,
    roomNumber TEXT,
    capacity INTEGER ,
    timeSlot TIME
);
INSERT INTO "Classes" VALUES(1,'Introduction to Computer Science','Dr. Johnson','Room 101',30,'9:00 AM - 11:00 AM');
INSERT INTO "Classes" VALUES(2,'Electronic Circuits','Prof. Smith','Room 202',28,'11:30 AM - 1:30 PM');
INSERT INTO "Classes" VALUES(3,'Fluid Mechanics','Dr. Brown','Room 303',35,'2:00 PM - 4:00 PM');
INSERT INTO "Classes" VALUES(4,'Structural Analysis','Prof. White','Room 404',32,'4:30 PM - 6:30 PM');
INSERT INTO "Classes" VALUES(5,'Material Science','Dr. Black','Room 505',27,'7:00 PM - 9:00 PM');
CREATE TABLE Enrollments (
    enrollmentId INTEGER PRIMARY KEY AUTOINCREMENT,
    studentId INTEGER,
    classId INTEGER,
    grade CHAR(2) ,
    FOREIGN KEY (studentId) REFERENCES Students(id),
    FOREIGN KEY (classId) REFERENCES Classes(classId)
);
INSERT INTO "Enrollments" VALUES(1,1,1,'A');
INSERT INTO "Enrollments" VALUES(2,2,2,'B+');
INSERT INTO "Enrollments" VALUES(3,3,3,'A-');
INSERT INTO "Enrollments" VALUES(4,4,4,'B');
INSERT INTO "Enrollments" VALUES(5,5,5,'A');
CREATE TABLE Projects (
    projectId INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    dueDate DATE,
    status TEXT 
);
INSERT INTO "Projects" VALUES(1,'Optimizing Algorithms','Research on improving sorting algorithms.','2023-05-15','Not Started');
INSERT INTO "Projects" VALUES(2,'Renewable Energy Storage','Exploring energy storage solutions for renewable sources.','2023-05-20','In Progress');
INSERT INTO "Projects" VALUES(3,'Urban Transportation Systems','Designing efficient transportation systems for urban areas.','2023-05-25','Completed');
INSERT INTO "Projects" VALUES(4,'Nanotechnology Applications','Investigating potential applications of nanotechnology in various fields.','2023-05-30','Pending Review');
INSERT INTO "Projects" VALUES(5,'Artificial Intelligence Ethics','Examining ethical considerations of AI and its impact on society.','2023-06-05','Not Started');
CREATE TABLE Students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    major TEXT,
    yearInSchool INTEGER 
);
INSERT INTO "Students" VALUES(1,'John','Doe','Computer Science',3);
INSERT INTO "Students" VALUES(2,'Jane','Smith','Electrical Engineering',2);
INSERT INTO "Students" VALUES(3,'Alice','Brown','Mechanical Engineering',4);
INSERT INTO "Students" VALUES(4,'Michael','White','Civil Engineering',3);
INSERT INTO "Students" VALUES(5,'Emily','Black','Aerospace Engineering',2);
CREATE TABLE Submissions (
    submissionId INTEGER PRIMARY KEY AUTOINCREMENT,
    studentId INTEGER,
    themeTrackId INTEGER,
    projectId INTEGER,
    dateOfSubmission DATETIME,
    feedback TEXT,
    grade CHAR(2) ,
    FOREIGN KEY (studentId) REFERENCES Students(id),
    FOREIGN KEY (themeTrackId) REFERENCES Theme_Tracks(themeTrackId),
    FOREIGN KEY (projectId) REFERENCES Projects(projectId)
);
INSERT INTO "Submissions" VALUES(1,1,1,1,'2023-04-15 10:00:00','Well done! Consider focusing more on practical implementation.','A');
INSERT INTO "Submissions" VALUES(2,2,2,2,'2023-04-20 10:00:00','Good job! More work is needed on the theoretical aspects.','B');
INSERT INTO "Submissions" VALUES(3,3,3,3,'2023-04-25 10:00:00','Great effort! The literature review could be improved further.','A-');
INSERT INTO "Submissions" VALUES(4,4,4,4,'2023-05-05 10:00:00','Needs improvement. Focus on addressing the problem statement more clearly.','C');
INSERT INTO "Submissions" VALUES(5,5,5,5,'2023-05-10 10:00:00','Very well written! Some minor revisions are suggested before final submission.','A');
CREATE TABLE "Theme_Tracks" (
    themeTrackId INTEGER PRIMARY KEY AUTOINCREMENT,
    Focus TEXT,
    conference TEXT,
    abstract TEXT,
    fullPaper TEXT
);
INSERT INTO "Theme_Tracks" VALUES(1,'Machine Learning','International Conference on Artificial Intelligence','This paper presents a novel approach to NLP...','The full paper can be found at https://example.com/fullpaper.pdf');
INSERT INTO "Theme_Tracks" VALUES(2,'Robotics','Annual Robotics Symposium','We propose an innovative swarm robotic system for monitoring...','https://example.com/fullpaper_robotics.pdf');
INSERT INTO "Theme_Tracks" VALUES(3,'Green Technologies','Global Climate Summit','Our research focuses on developing ML models for CCS...','https://example.com/fullpaper_green.pdf');
INSERT INTO "Theme_Tracks" VALUES(4,'Healthcare','World Health Congress','This study investigates the effectiveness of wearables in continuously monitoring...','https://example.com/fullpaper_healthcare.pdf');
INSERT INTO "Theme_Tracks" VALUES(5,'Space Exploration','Interplanetary Conference','This paper explores recent breakthroughs in propulsion technology...','https://example.com/fullpaper_space.pdf');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Students',5);
INSERT INTO "sqlite_sequence" VALUES('Classes',5);
INSERT INTO "sqlite_sequence" VALUES('Enrollments',5);
INSERT INTO "sqlite_sequence" VALUES('Projects',5);
INSERT INTO "sqlite_sequence" VALUES('Submissions',5);
INSERT INTO "sqlite_sequence" VALUES('Theme_Tracks',5);
COMMIT;
