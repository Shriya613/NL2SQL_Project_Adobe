BEGIN TRANSACTION;
CREATE TABLE Design_Class(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT,
    Skills TEXT,
    Projects TEXT
);
INSERT INTO "Design_Class" VALUES(1,'Eve',20,'Graphic Design, Photoshop','Sustainability Project');
INSERT INTO "Design_Class" VALUES(2,'Frank',22,'Illustrator, InDesign','Green Building Design');
INSERT INTO "Design_Class" VALUES(3,'Grace',21,'Photography, Lightroom','Urban Planning Proposal');
INSERT INTO "Design_Class" VALUES(4,'Hannah',23,'Web Design, HTML/CSS','Renewable Energy Solutions');
CREATE TABLE Engineering_Class(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT,
    Grade REAL,
    Projects TEXT
);
INSERT INTO "Engineering_Class" VALUES(1,'Alice',20,3.8,'Sustainability Project');
INSERT INTO "Engineering_Class" VALUES(2,'Bob',22,3.6,'Robotics Club Project');
INSERT INTO "Engineering_Class" VALUES(3,'Carol',21,3.7,'Smart City Initiative');
INSERT INTO "Engineering_Class" VALUES(4,'David',23,3.9,'Autonomous Vehicle Research');
CREATE TABLE Parents(
    id INTEGER PRIMARY KEY,
    FullName TEXT,
    PhoneNumber VARCHAR(15),
    Address TEXT
);
INSERT INTO "Parents" VALUES(1,'Tom Doe','555-123-4567','123 Main St.');
INSERT INTO "Parents" VALUES(2,'Anne Smith','555-987-6543','456 Oak Ave.');
INSERT INTO "Parents" VALUES(3,'Mark Johnson','555-555-5555','789 Pine St.');
INSERT INTO "Parents" VALUES(4,'Sarah Williams','555-111-2222','987 Elm St.');
INSERT INTO "Parents" VALUES(5,'Steve Davis','555-333-4444','135 Maple Dr.');
CREATE TABLE School(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    NumberOfStudents INT,
    Principal TEXT
);
INSERT INTO "School" VALUES(1,'Springfield High School','Springfield, IL',1200,'Principal White');
INSERT INTO "School" VALUES(2,'Washington Middle School','Springfield, IL',600,'Principal Taylor');
INSERT INTO "School" VALUES(3,'Lincoln Elementary School','Springfield, IL',300,'Principal Johnson');
INSERT INTO "School" VALUES(4,'Jefferson College','Chicago, IL',1500,'President Adams');
INSERT INTO "School" VALUES(5,'Madison University','Peoria, IL',2500,'Chancellor Jackson');
CREATE TABLE Students(
    id INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    Email TEXT UNIQUE
);
INSERT INTO "Students" VALUES(1,'John','Doe','2000-01-01','john.doe@example.com');
INSERT INTO "Students" VALUES(2,'Jane','Smith','2001-02-02','jane.smith@example.com');
INSERT INTO "Students" VALUES(3,'Jack','Johnson','2002-03-03','jack.johnson@example.com');
INSERT INTO "Students" VALUES(4,'Jim','Williams','2003-04-04','jim.williams@example.com');
INSERT INTO "Students" VALUES(5,'Jill','Davis','2004-05-05','jill.davis@example.com');
CREATE TABLE Subjects(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Teacher TEXT,
    Classroom TEXT
);
INSERT INTO "Subjects" VALUES(1,'Calculus','Mr. Smith','Room A101');
INSERT INTO "Subjects" VALUES(2,'Physics','Mrs. Johnson','Room B102');
INSERT INTO "Subjects" VALUES(3,'Chemistry','Dr. Brown','Room C103');
INSERT INTO "Subjects" VALUES(4,'Biology','Ms. Davis','Room D104');
INSERT INTO "Subjects" VALUES(5,'Computer Science','Professor Garcia','Lab E105');
COMMIT;
