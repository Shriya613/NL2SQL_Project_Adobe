BEGIN TRANSACTION;
CREATE TABLE Editorial_Teams(
    TeamID INTEGER PRIMARY KEY AUTOINCREMENT,
    TeamName TEXT,
    TeamDescription TEXT
);
INSERT INTO "Editorial_Teams" VALUES(1,'Team1','A team focused on sports journalism.');
INSERT INTO "Editorial_Teams" VALUES(2,'Team2','Specializes in political news coverage.');
INSERT INTO "Editorial_Teams" VALUES(3,'Team3','Focuses on technology and innovation articles.');
INSERT INTO "Editorial_Teams" VALUES(4,'Team4','Covering arts, culture, and entertainment stories.');
INSERT INTO "Editorial_Teams" VALUES(5,'Team5','Dedicated to investigative reporting.');
CREATE TABLE Editorial_Teams_Roles(
    TeamID INTEGER,
    RoleID INTEGER,
    FOREIGN KEY(TeamID) REFERENCES Editorial_Teams(TeamID),
    FOREIGN KEY(RoleID) REFERENCES Roles(RoleID),
    PRIMARY KEY(TeamID, RoleID)
);
INSERT INTO "Editorial_Teams_Roles" VALUES(1,1);
INSERT INTO "Editorial_Teams_Roles" VALUES(2,1);
INSERT INTO "Editorial_Teams_Roles" VALUES(3,1);
INSERT INTO "Editorial_Teams_Roles" VALUES(4,1);
INSERT INTO "Editorial_Teams_Roles" VALUES(5,1);
INSERT INTO "Editorial_Teams_Roles" VALUES(5,5);
CREATE TABLE Roles(
    RoleID INTEGER PRIMARY KEY AUTOINCREMENT,
    RoleName TEXT
);
INSERT INTO "Roles" VALUES(1,'Copy_Editor');
INSERT INTO "Roles" VALUES(2,'Reporter');
INSERT INTO "Roles" VALUES(3,'Photographer');
INSERT INTO "Roles" VALUES(4,'Designer');
INSERT INTO "Roles" VALUES(5,'Web Developer');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Editorial_Teams',5);
INSERT INTO "sqlite_sequence" VALUES('Roles',5);
COMMIT;
