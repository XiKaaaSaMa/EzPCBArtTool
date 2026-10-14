import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import com.jhlabs.image.AbstractBufferedImageOp;

/**
 * PCB 艺术板工具 · Java 滤镜小工具（独立子工具，供主程序或命令行调用）
 *
 * 用法：
 *   java -cp "classes;lib/*" LineArtJavaTool list
 *   java -cp "classes;lib/*" LineArtJavaTool filter <滤镜名> <输入图> <输出图>
 *
 * - filter：JH Labs Filters（Apache-2.0）滤镜，含线稿类
 *        （stamp/skeleton/edge/contour/dog/threshold）与艺术类
 *        （crystallize/marble/chrome/weave/pointillize/emboss/smear/diffuse/median/sharpen）；
 *        `null` = 不加滤镜，原样输出（与主程序下拉框的「无」对应）
 * - 输入图带透明通道时先按"直通 alpha"合成到白底（与主程序规则一致）。
 *
 * 源码与 jar 都随仓库分发；`classes/LineArtJavaTool.class` 是按 Java 8 目标
 * 预编译的产物，装了 JRE（无需 JDK）也能直接跑；改了源码要重新运行 build.cmd。
 */
public class LineArtJavaTool {

    private static final String[] FILTERS = {
        "null",
        "stamp", "skeleton", "edge", "contour", "dog", "threshold",
        "crystallize", "marble", "chrome", "weave", "pointillize", "emboss",
        "smear", "diffuse", "median", "sharpen"
    };

    public static void main(String[] args) {
        try {
            if (args.length == 0) { usage(); System.exit(2); }
            String cmd = args[0].toLowerCase();
            if (cmd.equals("list")) {
                for (String f : FILTERS) System.out.println(f);
            } else if (cmd.equals("filter")) {
                if (args.length < 4) { usage(); System.exit(2); }
                BufferedImage img = readFlat(args[2]);
                // null = 不加滤镜：直接原样输出（与主程序对话框的“不滤镜”对应）
                BufferedImage out = args[1].equalsIgnoreCase("null")
                        ? img : makeFilter(args[1]).filter(img, null);
                String fmt = args[3].toLowerCase().endsWith(".jpg")
                        || args[3].toLowerCase().endsWith(".jpeg") ? "jpeg" : "png";
                ImageIO.write(out, fmt, new File(args[3]));
                System.out.println("OK filter " + args[1] + " " + args[3]);
            } else {
                usage(); System.exit(2);
            }
        } catch (Exception e) {
            System.err.println("ERROR: " + e);
            System.exit(1);
        }
    }

    private static void usage() {
        System.out.println("用法: LineArtJavaTool list | filter <name> <in> <out>");
    }

    /**
     * 读取图片并压平透明通道（合成到白底）。
     *
     * ⚠️ **必须是 TYPE_INT_ARGB，不能是 TYPE_INT_RGB。** JH Labs 的滤镜按
     * 「图自带 alpha 通道」写：在 TYPE_INT_RGB 上，SharpenFilter 走的
     * ConvolveFilter 默认 premultiplyAlpha=true，取出的 alpha 恒为 0，
     * 预乘把每个像素乘成 0 —— **整幅输出全黑**（实测均值 0.0，换 ARGB 后
     * 127.98）；DoGFilter 的内部差分会变成常量图，归一化那步除零抛
     * ArithmeticException。两处实测症状见 java_lineart/README.md。
     * 换成 ARGB 且 alpha 恒为 255 之后，16 个滤镜全部正常。
     */
    private static BufferedImage readFlat(String path) throws Exception {
        BufferedImage src = ImageIO.read(new File(path));
        if (src == null) throw new Exception("无法读取图片: " + path);
        BufferedImage argb = new BufferedImage(src.getWidth(), src.getHeight(),
                                               BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = argb.createGraphics();
        g.drawImage(src, 0, 0, Color.WHITE, null);   // 透明区合成白底，alpha 落成 255
        g.dispose();
        return argb;
    }

    private static AbstractBufferedImageOp makeFilter(String name) throws Exception {
        switch (name.toLowerCase()) {
            case "stamp":       return new com.jhlabs.image.StampFilter();
            case "skeleton":    return new com.jhlabs.image.SkeletonFilter();
            case "edge":        return new com.jhlabs.image.EdgeFilter();
            case "contour":     return new com.jhlabs.image.ContourFilter();
            case "dog":         return new com.jhlabs.image.DoGFilter();
            case "threshold":   return new com.jhlabs.image.ThresholdFilter();
            case "crystallize": return new com.jhlabs.image.CrystallizeFilter();
            case "marble":      return new com.jhlabs.image.MarbleFilter();
            case "chrome":      return new com.jhlabs.image.ChromeFilter();
            case "weave":       return new com.jhlabs.image.WeaveFilter();
            case "pointillize": return new com.jhlabs.image.PointillizeFilter();
            case "emboss":      return new com.jhlabs.image.EmbossFilter();
            case "smear":       return new com.jhlabs.image.SmearFilter();
            case "diffuse":     return new com.jhlabs.image.DiffuseFilter();
            case "median":      return new com.jhlabs.image.MedianFilter();
            case "sharpen":     return new com.jhlabs.image.SharpenFilter();
            default: throw new Exception("未知滤镜: " + name + "（可用: " + String.join("/", FILTERS) + "）");
        }
    }
}
