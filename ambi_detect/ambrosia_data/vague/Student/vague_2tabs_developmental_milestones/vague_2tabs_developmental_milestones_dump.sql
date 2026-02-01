BEGIN TRANSACTION;
CREATE TABLE Child_Age (
    child_id INTEGER,
    physical_milestone_id INTEGER,
    cognitive_milestone_id INTEGER,
    FOREIGN KEY(child_id) REFERENCES "Children"(id),
    FOREIGN KEY(physical_milestone_id) REFERENCES Physical_Milestones(id),
    FOREIGN KEY(cognitive_milestone_id) REFERENCES Cognitive_Milestones(id)
);
INSERT INTO "Child_Age" VALUES(1,1,1);
INSERT INTO "Child_Age" VALUES(2,2,2);
INSERT INTO "Child_Age" VALUES(3,3,3);
INSERT INTO "Child_Age" VALUES(4,4,4);
INSERT INTO "Child_Age" VALUES(5,5,5);
CREATE TABLE "Children" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER
);
INSERT INTO "Children" VALUES(1,'Alice',2);
INSERT INTO "Children" VALUES(2,'Bob',3);
INSERT INTO "Children" VALUES(3,'Charlie',4);
INSERT INTO "Children" VALUES(4,'David',5);
INSERT INTO "Children" VALUES(5,'Eva',6);
CREATE TABLE Cognitive_Milestones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    milestone_name TEXT,
    description TEXT
);
INSERT INTO "Cognitive_Milestones" VALUES(1,'Recognition','Can recognize colors');
INSERT INTO "Cognitive_Milestones" VALUES(2,'Memory','Remembers own name');
INSERT INTO "Cognitive_Milestones" VALUES(3,'Problem Solving','Can solve simple problems');
INSERT INTO "Cognitive_Milestones" VALUES(4,'Understanding','Shows understanding of simple instructions');
INSERT INTO "Cognitive_Milestones" VALUES(5,'Imitation','Imitates actions of others');
CREATE TABLE Physical_Milestones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    milestone_name TEXT,
    description TEXT
);
INSERT INTO "Physical_Milestones" VALUES(1,'Walking','Can walk independently without support');
INSERT INTO "Physical_Milestones" VALUES(2,'Jumping','Can jump in place');
INSERT INTO "Physical_Milestones" VALUES(3,'Running','Can run for short distances');
INSERT INTO "Physical_Milestones" VALUES(4,'Climbing','Can climb onto and off of furniture without assistance');
INSERT INTO "Physical_Milestones" VALUES(5,'Hopping','Can hop on one foot');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Children',5);
INSERT INTO "sqlite_sequence" VALUES('Physical_Milestones',5);
INSERT INTO "sqlite_sequence" VALUES('Cognitive_Milestones',5);
COMMIT;
