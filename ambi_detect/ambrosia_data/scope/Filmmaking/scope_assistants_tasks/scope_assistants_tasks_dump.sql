BEGIN TRANSACTION;
CREATE TABLE Assistants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
);
INSERT INTO "Assistants" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Assistants" VALUES(2,'Jane Smith','jane@example.com');
INSERT INTO "Assistants" VALUES(3,'Michael Johnson','michael@example.com');
INSERT INTO "Assistants" VALUES(4,'Emily Davis','emily@example.com');
INSERT INTO "Assistants" VALUES(5,'David Brown','david@example.com');
CREATE TABLE Assistants_Tasks (
    assistant_id INTEGER,
    task_id INTEGER,
    FOREIGN KEY(assistant_id) REFERENCES Assistants(id),
    FOREIGN KEY(task_id) REFERENCES Tasks(id),
    PRIMARY KEY(assistant_id, task_id)
);
INSERT INTO "Assistants_Tasks" VALUES(1,2);
INSERT INTO "Assistants_Tasks" VALUES(2,2);
INSERT INTO "Assistants_Tasks" VALUES(3,2);
INSERT INTO "Assistants_Tasks" VALUES(4,2);
INSERT INTO "Assistants_Tasks" VALUES(5,2);
INSERT INTO "Assistants_Tasks" VALUES(5,1);
CREATE TABLE Tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT,
    description TEXT
);
INSERT INTO "Tasks" VALUES(1,'Location Scouting','Scout locations for shooting scenes.');
INSERT INTO "Tasks" VALUES(2,'Casting Calls','Organize casting calls for actors.');
INSERT INTO "Tasks" VALUES(3,'Props Acquisition','Acquire necessary props for sets.');
INSERT INTO "Tasks" VALUES(4,'Set Design','Design and plan set designs.');
INSERT INTO "Tasks" VALUES(5,'Budget Management','Manage film budget.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Assistants',5);
INSERT INTO "sqlite_sequence" VALUES('Tasks',5);
COMMIT;
