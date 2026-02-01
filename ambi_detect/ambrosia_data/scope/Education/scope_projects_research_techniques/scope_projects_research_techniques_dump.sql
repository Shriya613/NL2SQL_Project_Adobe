BEGIN TRANSACTION;
CREATE TABLE Projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name TEXT,
    description TEXT
);
INSERT INTO "Projects" VALUES(1,'Educational Game Development','Creating engaging educational games');
INSERT INTO "Projects" VALUES(2,'Online Learning Platform','Developing a platform for online learning');
INSERT INTO "Projects" VALUES(3,'Smart Classroom System','Implementing IoT devices in classrooms');
INSERT INTO "Projects" VALUES(4,'AI-Based Tutoring System','Designing an AI system for personalized tutoring');
INSERT INTO "Projects" VALUES(5,'Augmented Reality Textbooks','Creating AR textbooks for enhanced learning experience');
CREATE TABLE Projects_Research_Techniques (
    project_id INTEGER,
    technique_id INTEGER,
    FOREIGN KEY(project_id) REFERENCES Projects(id),
    FOREIGN KEY(technique_id) REFERENCES Research_Techniques(id),
    PRIMARY KEY(project_id, technique_id)
);
INSERT INTO "Projects_Research_Techniques" VALUES(1,1);
INSERT INTO "Projects_Research_Techniques" VALUES(2,1);
INSERT INTO "Projects_Research_Techniques" VALUES(3,1);
INSERT INTO "Projects_Research_Techniques" VALUES(4,1);
INSERT INTO "Projects_Research_Techniques" VALUES(5,1);
INSERT INTO "Projects_Research_Techniques" VALUES(5,2);
CREATE TABLE Research_Techniques (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    technique_name TEXT,
    description TEXT
);
INSERT INTO "Research_Techniques" VALUES(1,'Surveys','Conducting surveys among educators and students');
INSERT INTO "Research_Techniques" VALUES(2,'Case Studies','Analyzing specific examples in depth');
INSERT INTO "Research_Techniques" VALUES(3,'Literature Review','Reviewing existing research on the topic');
INSERT INTO "Research_Techniques" VALUES(4,'Focus Group Discussions','Organizing discussions among targeted groups');
INSERT INTO "Research_Techniques" VALUES(5,'Expert Interviews','Interviewing experts in the field');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Projects',5);
INSERT INTO "sqlite_sequence" VALUES('Research_Techniques',5);
COMMIT;
