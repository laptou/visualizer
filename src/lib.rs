//! library entry for visualizer
//! overview: exposes modules for audio capture, ui rendering, draw context, and app boilerplate.

pub mod audio;
pub mod cli;
pub mod gfx;
// re-export for macro without making users depend on lyon directly
pub use lyon_path as __vzr_lyon_path;
pub use lyon_geom as __vzr_lyon_geom;
/// macro to build lyon svg-style paths concisely
/// usage:
/// let p = svg_path! {
///     move_to(10.0, 10.0);
///     line_to(50.0, 10.0);
///     quad_to(80.0, 10.0, 80.0, 40.0);
///     cubic_to(60.0, 60.0, 20.0, 60.0, 10.0, 40.0);
///     arc_to(20.0, 10.0, 0.0, false, true, 90.0, 40.0);
///     close();
/// };
#[macro_export]
macro_rules! svg_path {
    ( $( $cmd:tt )* ) => {{
        let mut _svg = $crate::__vzr_lyon_path::Path::builder().with_svg();
        svg_path!(@inner _svg, $( $cmd )* );
        _svg.build()
    }};

    (@inner $svg:ident, move_to($x:expr, $y:expr); $( $rest:tt )* ) => {{
        $svg.move_to($crate::__vzr_lyon_geom::point($x as f32, $y as f32));
        svg_path!(@inner $svg, $( $rest )* );
    }};
    (@inner $svg:ident, line_to($x:expr, $y:expr); $( $rest:tt )* ) => {{
        $svg.line_to($crate::__vzr_lyon_geom::point($x as f32, $y as f32));
        svg_path!(@inner $svg, $( $rest )* );
    }};
    (@inner $svg:ident, quad_to($cx:expr, $cy:expr, $x:expr, $y:expr); $( $rest:tt )* ) => {{
        $svg.quadratic_bezier_to($crate::__vzr_lyon_geom::point($cx as f32, $cy as f32), $crate::__vzr_lyon_geom::point($x as f32, $y as f32));
        svg_path!(@inner $svg, $( $rest )* );
    }};
    (@inner $svg:ident, cubic_to($cx1:expr, $cy1:expr, $cx2:expr, $cy2:expr, $x:expr, $y:expr); $( $rest:tt )* ) => {{
        $svg.cubic_bezier_to(
            $crate::__vzr_lyon_geom::point($cx1 as f32, $cy1 as f32),
            $crate::__vzr_lyon_geom::point($cx2 as f32, $cy2 as f32),
            $crate::__vzr_lyon_geom::point($x as f32, $y as f32),
        );
        svg_path!(@inner $svg, $( $rest )* );
    }};
    (@inner $svg:ident, arc_to($rx:expr, $ry:expr, $xrot:expr, $large:expr, $sweep:expr, $x:expr, $y:expr); $( $rest:tt )* ) => {{
        $svg.arc_to(
            $crate::__vzr_lyon_geom::vector($rx as f32, $ry as f32),
            $xrot as f32,
            $large,
            $sweep,
            $crate::__vzr_lyon_geom::point($x as f32, $y as f32),
        );
        svg_path!(@inner $svg, $( $rest )* );
    }};
    (@inner $svg:ident, close(); $( $rest:tt )* ) => {{
        $svg.close();
        svg_path!(@inner $svg, $( $rest )* );
    }};
    (@inner $svg:ident, ) => {};
}
pub mod shared;
pub mod ui;
pub mod app;


