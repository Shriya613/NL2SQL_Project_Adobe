BEGIN TRANSACTION;
CREATE TABLE "Internship" (
    id INTEGER PRIMARY KEY,
    student_id INTEGER,
    industry_sector_id INTEGER,
    FOREIGN KEY(student_id) REFERENCES Students(id),
    FOREIGN KEY(industry_sector_id) REFERENCES "Topic"(id));
INSERT INTO "Internship" VALUES(1,1,1);
INSERT INTO "Internship" VALUES(2,2,2);
INSERT INTO "Internship" VALUES(3,3,3);
INSERT INTO "Internship" VALUES(4,4,4);
INSERT INTO "Internship" VALUES(5,5,5);
CREATE TABLE "Topic" (
    id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT
);
INSERT INTO "Topic" VALUES(1,'Technology','Includes companies involved in software development, hardware manufacturing and IT services.');
INSERT INTO "Topic" VALUES(2,'Finance','Companies dealing with financial transactions, investments and banking operations.');
INSERT INTO "Topic" VALUES(3,'Healthcare','Organizations providing medical care, research and pharmaceutical products.');
INSERT INTO "Topic" VALUES(4,'Education','Institutions focused on teaching, training and learning activities.');
INSERT INTO "Topic" VALUES(5,'Retail','Businesses that sell goods and services directly to consumers.');
COMMIT;
