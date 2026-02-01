BEGIN TRANSACTION;
CREATE TABLE Classroom(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    capacity INTEGER);
INSERT INTO "Classroom" VALUES(1,'Room A',30);
INSERT INTO "Classroom" VALUES(2,'Room B',25);
INSERT INTO "Classroom" VALUES(3,'Room C',28);
INSERT INTO "Classroom" VALUES(4,'Room D',32);
INSERT INTO "Classroom" VALUES(5,'Room E',35);
CREATE TABLE Classroom_Equipment(
    classroomId INTEGER,
    equipmentId INTEGER,
    FOREIGN KEY(classroomId) REFERENCES Classroom(id),
    FOREIGN KEY(equipmentId) REFERENCES Equipment(id));
INSERT INTO "Classroom_Equipment" VALUES(1,1);
INSERT INTO "Classroom_Equipment" VALUES(1,2);
INSERT INTO "Classroom_Equipment" VALUES(2,1);
INSERT INTO "Classroom_Equipment" VALUES(2,3);
INSERT INTO "Classroom_Equipment" VALUES(3,1);
INSERT INTO "Classroom_Equipment" VALUES(3,4);
INSERT INTO "Classroom_Equipment" VALUES(4,1);
INSERT INTO "Classroom_Equipment" VALUES(4,5);
INSERT INTO "Classroom_Equipment" VALUES(5,1);
CREATE TABLE Equipment(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    equipmentName TEXT,
    quantity INTEGER);
INSERT INTO "Equipment" VALUES(1,'Desks And Chairs',10);
INSERT INTO "Equipment" VALUES(2,'Smart Board',5);
INSERT INTO "Equipment" VALUES(3,'Projector Screen',6);
INSERT INTO "Equipment" VALUES(4,'Whiteboard',7);
INSERT INTO "Equipment" VALUES(5,'Computers',8);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Classroom',5);
INSERT INTO "sqlite_sequence" VALUES('Equipment',5);
COMMIT;
