SELECT model FROM solution
    JOIN bench AS nohelp
        ON nohelp.problem_id = solution.problem_id AND nohelp.help == 0.0
    JOIN bench AS yehelp
        ON yehelp.problem_id = solution.problem_id AND yehelp.help > 0.0
    WHERE nohelp.iteration = yehelp.iteration
        AND nohelp.runtime < yehelp.runtime
        AND model LIKE '%\u{%}%';
