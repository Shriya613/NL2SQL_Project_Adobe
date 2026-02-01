BEGIN TRANSACTION;
CREATE TABLE Articles(
    id INTEGER PRIMARY KEY,
    title TEXT,
    content TEXT,
    author_id INTEGER REFERENCES Authors(author_id),
    reporter_id INTEGER REFERENCES Reporters(reporter_id),
    final_draft_id INTEGER REFERENCES Final_Draft(id));
INSERT INTO "Articles" VALUES(1,'Breaking News Story','This is a news story about breaking news.',1,1,1);
INSERT INTO "Articles" VALUES(2,'Feature on Local Business','A feature article about a local business.',2,2,2);
CREATE TABLE Authors(
    author_id INTEGER PRIMARY KEY,
    fullname TEXT);
INSERT INTO "Authors" VALUES(1,'Alice Brown');
INSERT INTO "Authors" VALUES(2,'Charlie Green');
CREATE TABLE Copy_Editor(
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT UNIQUE);
INSERT INTO "Copy_Editor" VALUES(1,'Mike Johnson','mike@example.com');
INSERT INTO "Copy_Editor" VALUES(2,'Emily Davis','emily@example.com');
CREATE TABLE Final_Draft(
    id INTEGER PRIMARY KEY,
    managing_editor_id INTEGER REFERENCES Managing_Editor(id),
    copy_editor_id INTEGER REFERENCES Copy_Editor(id));
INSERT INTO "Final_Draft" VALUES(1,1,1);
INSERT INTO "Final_Draft" VALUES(2,2,2);
CREATE TABLE Managing_Editor(
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT UNIQUE);
INSERT INTO "Managing_Editor" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Managing_Editor" VALUES(2,'Jane Smith','jane@example.com');
CREATE TABLE Reporters(
    reporter_id INTEGER PRIMARY KEY,
    fullname TEXT);
INSERT INTO "Reporters" VALUES(1,'David Black');
INSERT INTO "Reporters" VALUES(2,'Eva White');
COMMIT;
