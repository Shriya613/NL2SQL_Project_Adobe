BEGIN TRANSACTION;
CREATE TABLE Interior_Designer (
    designer_id INTEGER PRIMARY KEY,
    name TEXT,
    design_style INTEGER REFERENCES Style(style_id)
);
INSERT INTO "Interior_Designer" VALUES(1,'John Doe',1);
INSERT INTO "Interior_Designer" VALUES(2,'Jane Smith',2);
INSERT INTO "Interior_Designer" VALUES(3,'Michael Johnson',3);
INSERT INTO "Interior_Designer" VALUES(4,'Emily Davis',4);
INSERT INTO "Interior_Designer" VALUES(5,'David Brown',5);
CREATE TABLE Landscape_Architect (
    architect_id INTEGER PRIMARY KEY,
    name TEXT,
    landscape_style INTEGER REFERENCES Style(style_id)
);
INSERT INTO "Landscape_Architect" VALUES(1,'Mark Williams',1);
INSERT INTO "Landscape_Architect" VALUES(2,'Laura Wilson',2);
INSERT INTO "Landscape_Architect" VALUES(3,'Thomas Jackson',3);
INSERT INTO "Landscape_Architect" VALUES(4,'Nicholas Taylor',4);
INSERT INTO "Landscape_Architect" VALUES(5,'William Martin',5);
CREATE TABLE Material (
    material_id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT
);
INSERT INTO "Material" VALUES(1,'Wood','High quality hardwood');
INSERT INTO "Material" VALUES(2,'Stone','Variety of natural stones');
INSERT INTO "Material" VALUES(3,'Glass','Tempered and laminated glass');
INSERT INTO "Material" VALUES(4,'Metal','Stainless steel and brass');
INSERT INTO "Material" VALUES(5,'Textile','Cotton, silk, and wool fabrics');
CREATE TABLE Project (
    project_id INTEGER PRIMARY KEY,
    title TEXT,
    start_date DATE ,
    end_date DATE ,
    interior_designer_id INTEGER,
    landscape_architect_id INTEGER,
    FOREIGN KEY (interior_designer_id) REFERENCES Interior_Designer(designer_id),
    FOREIGN KEY (landscape_architect_id) REFERENCES Landscape_Architect(architect_id)
);
INSERT INTO "Project" VALUES(1,'Living Room Redesign','2021-07-01','2021-09-01',1,1);
INSERT INTO "Project" VALUES(2,'Kitchen Remodel','2021-08-01','2021-10-01',2,2);
INSERT INTO "Project" VALUES(3,'Bathroom Renovation','2021-06-01','2021-08-01',3,3);
INSERT INTO "Project" VALUES(4,'Garden Design','2021-05-01','2021-07-01',4,4);
INSERT INTO "Project" VALUES(5,'Office Space Makeover','2021-04-01','2021-06-01',5,5);
CREATE TABLE ProjectMaterial (
    projectmaterial_id INTEGER PRIMARY KEY,
    project_id INTEGER,
    material_id INTEGER,
    quantity INTEGER ,
    FOREIGN KEY (project_id) REFERENCES Project(project_id),
    FOREIGN KEY (material_id) REFERENCES Material(material_id)
);
INSERT INTO "ProjectMaterial" VALUES(1,1,1,50);
INSERT INTO "ProjectMaterial" VALUES(2,1,2,25);
INSERT INTO "ProjectMaterial" VALUES(3,1,3,10);
INSERT INTO "ProjectMaterial" VALUES(4,1,4,20);
INSERT INTO "ProjectMaterial" VALUES(5,1,5,30);
CREATE TABLE Style (
    style_id INTEGER PRIMARY KEY,
    style_name TEXT UNIQUE
);
INSERT INTO "Style" VALUES(1,'Modern');
INSERT INTO "Style" VALUES(2,'Contemporary');
INSERT INTO "Style" VALUES(3,'Traditional');
INSERT INTO "Style" VALUES(4,'Rustic');
INSERT INTO "Style" VALUES(5,'Minimalist');
COMMIT;
