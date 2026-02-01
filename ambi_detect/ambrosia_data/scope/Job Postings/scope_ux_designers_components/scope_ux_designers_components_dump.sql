BEGIN TRANSACTION;
CREATE TABLE Components (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    component_name TEXT,
    description TEXT
);
INSERT INTO "Components" VALUES(1,'Navigation','A navigation system helps users move between different sections of a website.');
INSERT INTO "Components" VALUES(2,'User Interface Design','UI design involves creating interfaces that are visually appealing and user-friendly.');
INSERT INTO "Components" VALUES(3,'Responsive Web Design','RWD ensures that websites function well on various devices and screen sizes.');
INSERT INTO "Components" VALUES(4,'Interaction Design','ID focuses on designing interactive elements within a UI to create an engaging experience.');
INSERT INTO "Components" VALUES(5,'Usability Testing','Testing how easy it is for users to complete tasks on a website.');
CREATE TABLE Ux_Designers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    years_of_experience INT 
);
INSERT INTO "Ux_Designers" VALUES(1,'John Doe',3);
INSERT INTO "Ux_Designers" VALUES(2,'Jane Smith',2);
INSERT INTO "Ux_Designers" VALUES(3,'Mike Johnson',6);
INSERT INTO "Ux_Designers" VALUES(4,'Emily Davis',1);
INSERT INTO "Ux_Designers" VALUES(5,'David Brown',7);
CREATE TABLE Ux_Designers_Components (
    ux_designer_id INTEGER,
    component_id INTEGER,
    FOREIGN KEY (ux_designer_id) REFERENCES Ux_Designers (id),
    FOREIGN KEY (component_id) REFERENCES Components (id)
);
INSERT INTO "Ux_Designers_Components" VALUES(1,1);
INSERT INTO "Ux_Designers_Components" VALUES(2,1);
INSERT INTO "Ux_Designers_Components" VALUES(3,1);
INSERT INTO "Ux_Designers_Components" VALUES(4,1);
INSERT INTO "Ux_Designers_Components" VALUES(5,1);
INSERT INTO "Ux_Designers_Components" VALUES(5,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Ux_Designers',5);
INSERT INTO "sqlite_sequence" VALUES('Components',5);
COMMIT;
