BEGIN TRANSACTION;
CREATE TABLE Graphic_Designers (
    id INTEGER PRIMARY KEY,
    officeId INTEGER,
    name TEXT,
    skills TEXT,
    FOREIGN KEY(officeId) REFERENCES Office(id)
);
INSERT INTO "Graphic_Designers" VALUES(1,1,'Alice Smith','Photoshop, Illustrator, InDesign');
INSERT INTO "Graphic_Designers" VALUES(2,2,'Bob Johnson','Photoshop, Illustrator, InDesign');
CREATE TABLE Office (
    id INTEGER PRIMARY KEY,
    address TEXT UNIQUE
);
INSERT INTO "Office" VALUES(1,'2/7 London Street');
INSERT INTO "Office" VALUES(2,'3/10 New York Avenue');
CREATE TABLE Web_Developers (
    id INTEGER PRIMARY KEY,
    officeId INTEGER,
    name TEXT,
    languages TEXT,
    FOREIGN KEY(officeId) REFERENCES Office(id)
);
INSERT INTO "Web_Developers" VALUES(1,1,'Carol Williams','JavaScript, HTML, CSS');
INSERT INTO "Web_Developers" VALUES(2,2,'David Brown','Python, Java, C++');
COMMIT;
